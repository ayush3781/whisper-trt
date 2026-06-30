# -*- coding: utf-8 -*-
import json
import re
import time
import traceback

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack

from .fbank import FeatureExtractor
from .tokenizer import LANGUAGES, get_tokenizer


class TritonPythonModel:
    """Triton Python backend model."""

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        self.tokenizer = get_tokenizer(num_languages=100)
        self.eos = self.tokenizer.encode(
            "<|endoftext|>",
            allowed_special=self.tokenizer.special_tokens_set,
        )[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)
        self.logger = pb_utils.Logger
        self.request_count = 0
        self.language_token_ids = None
        self.sot_id = self.tokenizer.encode(
            "<|startoftranscript|>",
            allowed_special=self.tokenizer.special_tokens_set,
        )[0]
        self._get_language_token_ids()
        self.init_model(self.model_config["parameters"])

    def init_model(self, parameters):
        for key, value in parameters.items():
            parameters[key] = value["string_value"]
        n_mels = int(parameters["n_mels"])
        self.zero_pad = True if parameters["zero_pad"] == "true" else False
        self.feature_extractor = FeatureExtractor(n_mels=n_mels)

    def _get_language_token_ids(self):
        if self.language_token_ids is None:
            self.language_token_ids = {}
            for lang in LANGUAGES:
                token = f"<|{lang}|>"
                try:
                    token_id = self.tokenizer.encode(
                        token,
                        allowed_special=self.tokenizer.special_tokens_set,
                    )[0]
                except Exception:
                    continue
                self.language_token_ids[lang] = token_id
        return self.language_token_ids

    def _extract_language_candidates(self, text_prefix):
        language_token_ids = self._get_language_token_ids()
        candidates = []
        seen = set()
        for match in re.finditer(r"<\|([a-z]{2,3})\|>", text_prefix):
            lang = match.group(1)
            if lang in language_token_ids and lang not in seen:
                candidates.append(lang)
                seen.add(lang)
        return candidates

    def _extract_prompt_context(self, text_prefix):
        return re.sub(r"^(?:\s*<\|[^|]+?\|>)*\s*", "", text_prefix).strip()

    def _build_text_prefix(self, lang, prompt_context=""):
        text_prefix = (
            f"<|startoftranscript|><|{lang}|>"
            "<|transcribe|><|notimestamps|>"
        )
        if prompt_context:
            text_prefix = f"{text_prefix} {prompt_context.strip()}"
        return text_prefix

    def _prepare_inputs(self, request, mel_feature, mel_len, prompt, max_tokens=256, return_log_probs=True):
        input_dict = {
            "request_output_len": np.array([[max_tokens]], dtype=np.int32),
            "end_id": np.array([[self.eos]], dtype=np.int32),
            "pad_id": np.array([[self.eos]], dtype=np.int32),
            "encoder_output_lengths": mel_len // 2,
            "input_lengths": mel_len,
            "decoder_input_ids": prompt,
            "streaming": np.array([[self.decoupled]], dtype=np.bool_),
            "return_log_probs": np.array([[return_log_probs]], dtype=np.bool_),
        }
        input_tensor_list = [pb_utils.Tensor(k, v) for k, v in input_dict.items()]
        input_tensor_list.append(
            pb_utils.Tensor.from_dlpack("encoder_input_features", to_dlpack(mel_feature.contiguous()))
        )
        return input_tensor_list

    def _detect_language(self, request, mel_feature, mel_len, candidate_langs=None, request_id=None):
        """Detect language from next-token probabilities after SOT."""
        language_token_ids = self._get_language_token_ids()
        if candidate_langs:
            candidates = [
                lang for lang in candidate_langs
                if lang in language_token_ids
            ]
        else:
            candidates = list(language_token_ids.keys())
        if not candidates:
            candidates = list(language_token_ids.keys())

        detect_prompt = np.array([[self.sot_id]], dtype=np.int32)
        detect_inputs = self._prepare_inputs(
            request,
            mel_feature,
            mel_len,
            detect_prompt,
            max_tokens=1,
            return_log_probs=False,
        )
        detect_inputs.append(
            pb_utils.Tensor("return_context_logits", np.array([[True]], dtype=np.bool_))
        )

        detect_request = pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=[
                "output_ids",
                "sequence_length",
                "context_logits",
            ],
            inputs=detect_inputs,
        )
        detect_start = time.perf_counter()
        detect_response = detect_request.exec(decoupled=False)
        detect_ms = (time.perf_counter() - detect_start) * 1000.0

        if detect_response.has_error():
            raise pb_utils.TritonModelException(detect_response.error().message())

        output_token_ids_full = pb_utils.get_output_tensor_by_name(
            detect_response, "output_ids"
        ).as_numpy().flatten().tolist()
        generated_token_ids = output_token_ids_full[-1:] if output_token_ids_full else []

        generated_lang = None
        generated_token = ""
        if generated_token_ids:
            generated_token = self.tokenizer.decode([generated_token_ids[0]]).strip()
            match = re.fullmatch(r"<\|([a-z]{2,3})\|>", generated_token)
            if match and match.group(1) in candidates:
                generated_lang = match.group(1)

        context_logits_tensor = pb_utils.get_output_tensor_by_name(
            detect_response,
            "context_logits",
        )
        if context_logits_tensor is None:
            selected_lang = generated_lang or ("en" if "en" in candidates else candidates[0])
            self.logger.log_info(
                f"whisper_timing request_id={request_id} "
                f"language={selected_lang} lang_detect_ms={detect_ms:.3f}"
            )
            return selected_lang

        context_logits = context_logits_tensor.as_numpy()
        next_token_logits = context_logits.reshape(-1, context_logits.shape[-1])[-1]
        scored_candidates = [
            (lang, float(next_token_logits[language_token_ids[lang]]))
            for lang in candidates
        ]
        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        selected_lang = scored_candidates[0][0]

        self.logger.log_info(
            f"whisper_timing request_id={request_id} "
            f"language={selected_lang} lang_detect_ms={detect_ms:.3f}"
        )

        return selected_lang

    def _prepare_llm_response(self, llm_request_inputs, request_id, transcript_start):
        llm_request = pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=["output_ids", "sequence_length", "output_log_probs", "cum_log_probs"],
            inputs=llm_request_inputs,
        )
        responses = llm_request.exec(decoupled=self.decoupled)

        # ============== NON-DECOUPLED MODE ==============
        if not self.decoupled:
            llm_response = responses
            if llm_response.has_error():
                raise pb_utils.TritonModelException(llm_response.error().message())

            output_token_ids_full = pb_utils.get_output_tensor_by_name(
                llm_response, "output_ids").as_numpy().flatten().tolist()
            output_log_probs_full = pb_utils.get_output_tensor_by_name(
                llm_response, "output_log_probs").as_numpy().flatten().tolist()
            cum_log_probs = pb_utils.get_output_tensor_by_name(
                llm_response, "cum_log_probs").as_numpy().flatten()

            num_generated = len(output_log_probs_full)
            output_token_ids = output_token_ids_full[-num_generated:] if num_generated > 0 else []

            output_text = self.tokenizer.decode(output_token_ids).strip()
            output_text = re.sub(r"<\|.*?\|>", "", output_text)

            transcript_ms = (time.perf_counter() - transcript_start) * 1000.0
            self.logger.log_info(
                f"whisper_timing request_id={request_id} "
                f"transcript_gen_ms={transcript_ms:.3f} transcript={output_text!r}"
            )

            output_token_ids_array = np.array(output_token_ids, dtype=np.int32)
            output_log_probs_array = np.array(output_log_probs_full, dtype=np.float32)
            cum_log_probs_array = np.array(cum_log_probs, dtype=np.float32)

            output_tensors = [
                pb_utils.Tensor("TRANSCRIPTS", np.array([output_text], dtype=np.object_)),
                pb_utils.Tensor("OUTPUT_TOKEN_IDS", output_token_ids_array),
                pb_utils.Tensor("CUM_LOG_PROBS", np.expand_dims(cum_log_probs_array, 0)),
                pb_utils.Tensor("OUTPUT_LOG_PROBS", np.expand_dims(output_log_probs_array, 0)),
            ]

            response = pb_utils.InferenceResponse(output_tensors)
            yield response

        # ============== DECOUPLED STREAMING MODE ==============
        else:
            output_token_ids_full = []
            output_log_probs_full = []
            cum_log_probs_list = []

            for llm_response in responses:
                if llm_response.has_error():
                    raise pb_utils.TritonModelException(llm_response.error().message())

                stream_output_ids = pb_utils.get_output_tensor_by_name(
                    llm_response, "output_ids").as_numpy().flatten().tolist()
                stream_log_probs = pb_utils.get_output_tensor_by_name(
                    llm_response, "output_log_probs").as_numpy().flatten().tolist()

                if not stream_output_ids:
                    continue

                output_token_ids_full.extend(stream_output_ids)
                output_log_probs_full.extend(stream_log_probs)

                try:
                    stream_cum_log_probs = pb_utils.get_output_tensor_by_name(
                        llm_response, "cum_log_probs")
                    if stream_cum_log_probs is not None:
                        cum_log_probs_list.append(stream_cum_log_probs.as_numpy())
                except Exception:
                    pass

                num_generated = len(output_log_probs_full)
                output_token_ids = output_token_ids_full[-num_generated:] if num_generated > 0 else []

                output_text = self.tokenizer.decode(output_token_ids).strip()
                output_text = re.sub(r"<\|.*?\|>", "", output_text)

                transcript_ms = (time.perf_counter() - transcript_start) * 1000.0
                self.logger.log_info(
                    f"whisper_timing request_id={request_id} "
                    f"transcript_gen_ms={transcript_ms:.3f} transcript={output_text!r}"
                )

                output_token_ids_array = np.array(output_token_ids, dtype=np.int32)
                output_log_probs_array = np.array(output_log_probs_full, dtype=np.float32)

                output_tensors = [
                    pb_utils.Tensor("TRANSCRIPTS", np.array([output_text], dtype=np.object_)),
                    pb_utils.Tensor("OUTPUT_TOKEN_IDS", output_token_ids_array),
                    pb_utils.Tensor("OUTPUT_LOG_PROBS", np.expand_dims(output_log_probs_array, 0)),
                ]

                if cum_log_probs_list:
                    cum_log_probs_array = np.concatenate(cum_log_probs_list, axis=0).astype(np.float32)
                    output_tensors.append(
                        pb_utils.Tensor("CUM_LOG_PROBS", np.expand_dims(cum_log_probs_array.flatten(), 0))
                    )

                response = pb_utils.InferenceResponse(output_tensors=output_tensors)
                yield response

    def execute(self, requests):
        responses = []
        for request in requests:
            self.request_count += 1
            request_id = self.request_count
            request_start = time.perf_counter()

            decoder_text_prompt = pb_utils.get_input_tensor_by_name(request, "TEXT_PREFIX").as_numpy().tolist()
            text_prefix = decoder_text_prompt[0][0].decode("utf-8")

            wav = pb_utils.get_input_tensor_by_name(request, "WAV").as_numpy()
            assert wav.shape[0] == 1, "Only support batch size 1"
            wav = torch.from_numpy(wav[0]).to(self.device)
            wav_len = pb_utils.get_input_tensor_by_name(request, "WAV_LENS").as_numpy().item()

            if self.zero_pad:
                wav = wav[:wav_len]
                target = 0
            else:
                target = 3000

            mel = self.feature_extractor.compute_feature(wav, target).transpose(1, 2)
            mel_len = np.array([[mel.shape[1]]], dtype=np.int32)

            text_prefix = text_prefix.strip()
            language_candidates = self._extract_language_candidates(text_prefix)
            prompt_context = self._extract_prompt_context(text_prefix)

            if text_prefix == "":
                detected_lang = self._detect_language(
                    request,
                    mel,
                    mel_len,
                    request_id=request_id,
                )
                text_prefix = self._build_text_prefix(detected_lang)
            elif len(language_candidates) > 1:
                detected_lang = self._detect_language(
                    request,
                    mel,
                    mel_len,
                    candidate_langs=language_candidates,
                    request_id=request_id,
                )
                text_prefix = self._build_text_prefix(detected_lang, prompt_context)
            elif (
                text_prefix.startswith("<|startoftranscript|>")
                and text_prefix != "<|startoftranscript|>"
                and not language_candidates
            ):
                detected_lang = self._detect_language(
                    request,
                    mel,
                    mel_len,
                    request_id=request_id,
                )
                text_prefix = self._build_text_prefix(detected_lang, prompt_context)

            prompt_id = self.tokenizer.encode(
                text_prefix,
                allowed_special=self.tokenizer.special_tokens_set,
            )
            decoder_input_ids = np.array([prompt_id], dtype=np.int32)

            if self.decoupled:
                response_sender = request.get_response_sender()

            try:
                llm_request_inputs = self._prepare_inputs(request, mel, mel_len, decoder_input_ids)
                if isinstance(llm_request_inputs, pb_utils.TritonError):
                    error = pb_utils.InferenceResponse(error=llm_request_inputs)
                    if self.decoupled:
                        response_sender.send(error, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                    else:
                        responses.append(error)

                transcript_start = time.perf_counter()
                llm_responses = self._prepare_llm_response(
                    llm_request_inputs,
                    request_id,
                    transcript_start,
                )

                for triton_response in llm_responses:
                    if self.decoupled:
                        response_sender.send(triton_response)
                    else:
                        responses.append(triton_response)

                total_ms = (time.perf_counter() - request_start) * 1000.0
                self.logger.log_info(
                    f"whisper_timing request_id={request_id} total_ms={total_ms:.3f}"
                )

                if self.decoupled:
                    response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

            except Exception:
                self.logger.log_error(traceback.format_exc())
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(traceback.format_exc())
                )
                if self.decoupled:
                    response_sender.send(error_response)
                    response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
                else:
                    responses.append(error_response)

        if self.decoupled:
            return None
        else:
            assert len(responses) == len(requests)
            return responses
