# -*- coding: utf-8 -*-
import json
import re
import traceback

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack

from .fbank import FeatureExtractor
from .tokenizer import get_tokenizer


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])

        self.tokenizer = get_tokenizer(num_languages=100)
        self.eos = self.tokenizer.encode("<|endoftext|>", allowed_special=self.tokenizer.special_tokens_set)[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config)
        self.logger = pb_utils.Logger
        self.init_model(self.model_config['parameters'])

    def init_model(self, parameters):
        for key, value in parameters.items():
            parameters[key] = value["string_value"]
        n_mels = int(parameters["n_mels"])
        self.zero_pad = True if parameters["zero_pad"] == "true" else False
        self.feature_extractor = FeatureExtractor(n_mels=n_mels)

    def _prepare_inputs(self, request, mel_feature, mel_len, prompt, max_tokens=256):
        input_dict = {
            "request_output_len": np.array([[max_tokens]], dtype=np.int32),
            "end_id": np.array([[self.eos]], dtype=np.int32),
            "pad_id": np.array([[self.eos]], dtype=np.int32),
            "encoder_output_lengths": mel_len // 2,
            "input_lengths": mel_len,
            "decoder_input_ids": prompt,
            "streaming": np.array([[self.decoupled]], dtype=np.bool_),
            "return_log_probs": np.array([[True]], dtype=np.bool_),
        }
        input_tensor_list = [pb_utils.Tensor(k, v) for k, v in input_dict.items()]
        input_tensor_list.append(
            pb_utils.Tensor.from_dlpack("encoder_input_features", to_dlpack(mel_feature.contiguous()))
        )
        return input_tensor_list

    def _detect_language(self, request, mel_feature, mel_len):
        """Use Whisper's own decoder to predict the language token.

        This mirrors Whisper auto-detect at the BLS level:
        1. Start decoding with only <|startoftranscript|>.
        2. Ask the decoder for one token.
        3. Treat that generated special token as the detected language.
        """
        sot_id = self.tokenizer.encode(
            "<|startoftranscript|>",
            allowed_special=self.tokenizer.special_tokens_set,
        )[0]

        detect_prompt = np.array([[sot_id]], dtype=np.int32)
        detect_inputs = self._prepare_inputs(
            request,
            mel_feature,
            mel_len,
            detect_prompt,
            max_tokens=1,
        )

        detect_request = pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=["output_ids", "sequence_length", "output_log_probs"],
            inputs=detect_inputs,
        )
        detect_response = detect_request.exec(decoupled=False)

        if detect_response.has_error():
            raise pb_utils.TritonModelException(detect_response.error().message())

        output_token_ids_full = pb_utils.get_output_tensor_by_name(
            detect_response, "output_ids"
        ).as_numpy().flatten().tolist()

        output_log_probs = pb_utils.get_output_tensor_by_name(
            detect_response, "output_log_probs"
        ).as_numpy().flatten().tolist()

        # The TRT-LLM output can include the prompt depending on config.
        # output_log_probs corresponds only to generated tokens, so use it
        # to slice out the generated part.
        num_generated = len(output_log_probs)
        generated_token_ids = (
            output_token_ids_full[-num_generated:]
            if num_generated > 0
            else output_token_ids_full[-1:]
        )

        if not generated_token_ids:
            return "en"

        detected_token = self.tokenizer.decode([generated_token_ids[0]]).strip()

        # Expected output: <|en|>, <|es|>, <|ar|>, etc.
        match = re.fullmatch(r"<\|([a-z]{2,3})\|>", detected_token)
        if match:
            return match.group(1)

        # Safe fallback if the decoder emits a non-language token.
        return "en"

    def _prepare_llm_response(self, llm_request_inputs):
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

            # Keep only generated tokens (exclude prompt)
            num_generated = len(output_log_probs_full)
            output_token_ids = output_token_ids_full[-num_generated:] if num_generated > 0 else []

            # Decode generated text
            output_text = self.tokenizer.decode(output_token_ids).strip()
            output_text = re.sub(r'<\|.*?\|>', '', output_text)

            # Convert to tensors
            output_token_ids_array = np.array(output_token_ids, dtype=np.int32)
            output_log_probs_array = np.array(output_log_probs_full, dtype=np.float32)
            cum_log_probs_array = np.array(cum_log_probs, dtype=np.float32)

            # Build response
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

                # Keep only generated tokens (exclude prompt)
                num_generated = len(output_log_probs_full)
                output_token_ids = output_token_ids_full[-num_generated:] if num_generated > 0 else []

                # Decode generated text
                output_text = self.tokenizer.decode(output_token_ids).strip()
                output_text = re.sub(r'<\|.*?\|>', '', output_text)

                # Convert to tensors
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
            decoder_text_prompt = pb_utils.get_input_tensor_by_name(request, "TEXT_PREFIX").as_numpy().tolist()
            text_prefix = decoder_text_prompt[0][0].decode('utf-8')

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

            # Low-latency auto-language path.
            # Do NOT call _detect_language() here, because that creates a
            # separate TRT-LLM request and turns auto-language into a two-pass flow.
            #
            # Empty TEXT_PREFIX => single-pass auto:
            #   <|startoftranscript|> -> model generates <|lang|>, task/control tokens, transcript
            #
            # If the caller sends a Whisper prefix that has task/control tokens but no
            # language token, normalize it to SOT-only as well. This avoids the bad case:
            #   <|startoftranscript|><|transcribe|><|notimestamps|>
            # where language is absent but auto-detect is not triggered.
            text_prefix = text_prefix.strip()
            has_lang_tag = re.search(r"<\|[a-z]{2,3}\|>", text_prefix) is not None

            if text_prefix == "":
                text_prefix = "<|startoftranscript|>"
            elif (
                text_prefix.startswith("<|startoftranscript|>")
                and text_prefix != "<|startoftranscript|>"
                and not has_lang_tag
            ):
                text_prefix = "<|startoftranscript|>"

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

                llm_responses = self._prepare_llm_response(llm_request_inputs)

                for triton_response in llm_responses:
                    if self.decoupled:
                        response_sender.send(triton_response)
                    else:
                        responses.append(triton_response)

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
