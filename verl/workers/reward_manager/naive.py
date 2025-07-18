# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from nltk.translate.bleu_score import sentence_bleu


def diversity_score(responses: list[str], response_ids: torch.Tensor, prompts: list[str]) -> torch.Tensor:
    """
    Compute the diversity score for a batch of responses.
    """
    # Get the unique indices
    prompt_to_indices = defaultdict(list)
    for idx, prompt in enumerate(prompts):
        prompt_to_indices[prompt].append(idx)
    
    # Calculate the diversity score
    index_to_rows = {prompt: [] for prompt in prompts}
    index_to_pos = {prompt: [] for prompt in prompts}
    index_to_set = {prompt: set() for prompt in prompts}

    for i, prompt in enumerate(prompts):
        max_tokens = min(15, response_ids[i].shape[0])
        index_to_rows[prompt].append(responses[i])
        index_to_pos[prompt].append(i)  # to maintain position of each rollout
        index_to_set[prompt].add(tuple(response_ids[i].tolist()[:max_tokens]))
        
    print("how many rollouts ", len(index_to_rows.keys()) , len(index_to_rows[prompt]))
    self_bleu_scores = [0.0]*len(response_ids)
    per_rollout_uniqueness = [0.0] * len(response_ids)  # aligned with input
    
    for i, response_id in enumerate(response_ids):
        max_tokens = min(15, response_id.shape[0])
        current_rollout_set = set(response_id.tolist()[:max_tokens])
        prompt = prompts[i]
        total_rollout_set = index_to_set[prompt] 
        #get the no. of tokens unique to current rollout compared to total rollout set
        unique_tokens = current_rollout_set - total_rollout_set
        #get the no. of tokens unique to total rollout set compared to current rollout set
        per_rollout_uniqueness[i] = len(unique_tokens) / len(current_rollout_set)
        
    for prompt in prompts:
        rollouts = index_to_rows[prompt]
        
        # Self-BLEU
        i=0
        if len(rollouts) < 2:
            self_bleu_scores[index_to_pos[prompt][i]] = 0.0
        else:
            bleu_scores = []
            for i in range(len(rollouts)):
                references = rollouts[:i] + rollouts[i+1:]
                score = sentence_bleu(references, rollouts[i])
                self_bleu_scores[index_to_pos[prompt][i]] = score
    
    return self_bleu_scores, per_rollout_uniqueness


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        indices = [item["index"] for item in data.non_tensor_batch["extra_info"]]
        print("indices ", indices)
        prompts = [item["prompt"] for item in data.non_tensor_batch["extra_info"]]
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        
        responses = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses.append(response_str.split(" "))

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
        
        self_bleu_scores, per_rollout_uniqueness = diversity_score(responses, data.batch["responses"], prompts)
        
        reward_extra_info["self_bleu_scores"] = self_bleu_scores
        reward_extra_info["per_rollout_uniqueness"] = per_rollout_uniqueness

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
