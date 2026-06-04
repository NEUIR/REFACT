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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    print(f"[DEBUG] default_compute_score called with data_source: '{data_source}'")
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "lighteval/MATH",
        "DigitalLearningGmbH/MATH-lighteval",
    ]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        # from . import math_dapo

        # res = math_dapo.compute_score(solution_str, ground_truth)
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
        "Big-Math-RL-Verified",
        "NuminaMath-1.5",
        "Omni-MATH-1",
        "TAL-SCQ5K",
        "mb_gaokao/gaokao2024_1122",
        "mb_gaokao/gaokaotagging2024_1123",
        "mb_gaokao/gaokaotagging2024_1126",
        "mb_gaokao/gaokaotagging2024_1127",
        "mb_gaokao/gaokaotagging2024_1128",
        "mb_gaokao/gaokaotagging2024_1129",
        "A-Chinese-Character-Puzzles-Dataset",
        "CC-Riddle",
        "deepscaler",
        "orz_hard",
        "orz_57k",
        "orz_72k_ext",
        "AREAL-RL",
        "AI-MO/NuminaMath-1.5",
        "Mix-Math",
        "FreedomIntelligence/medical-o1-verifiable-problem",
        "agentica-org/DeepScaleR-Preview-Dataset",
        "ink-usc/riddle_sense",
        "camel-ai/chemistry",
        "crawl/text_book",
        "camel-ai/physics",
        "camel-ai/biology",
        "Multi-subject-RLVR",
        "Math-RLVR",
        "AM-Math",
        "DAPO",
        "LIMO",
        "LIMR",
        "math500",
        "NuminaMath",
        "aime",
        "MetaMathQA",
        "openR1Math_extended",
        "data_ablation_full59K",
        "BigMathVerified",
        "ttrl_checked",
    ]:
        # from . import prime_math
        # res = prime_math.compute_score(solution_str, ground_truth)
        from deepscaler.rewards.math_reward import deepscaler_reward_fn

        res = deepscaler_reward_fn(solution_str, ground_truth)
    elif data_source in [
        "codecontests",
        "apps",
        "codeforces",
        "taco",
        "code_contests",
        "CODEFORCES",
        "likaixin/TACO-verified",
        "LeetCodeDataset",
    ]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url,
                concurrent_semaphore,
                solution_str,
                ground_truth,
                continuous=True,
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            # from . import prime_code

            # # Assuming prime_code doesn't need the URL
            # res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
            from . import coder1

            res = coder1.compute_score(
                solution_str,
                ground_truth,
                extra_info=extra_info,
                debug=False,
                format_reward=0.0,
                answer_reward=1.0,
            )
    elif data_source in [
        "code",
        "kodcode",
        "Algorithm",
        "Apps",
        "Code_Contests",
        "Codeforces",
        "Data_Structure",
        "Docs: Flask",
        "Docs: Pandas",
        "Docs: Python310",
        "Docs: Pytorch",
        "Docs: Scikit",
        "Docs: Seaborn",
        "Filter",
        "Leetcode",
        "Package",
        "Prefill",
        "Taco",
        "dedup_dataset_code",
        "matrixstudio/codeforces-python-submissions",
        "deepmind/code_contests",
        "crawl/icpc",
        "PrimeIntellect/SYNTHETIC-1",
        "darkbzoj",
        "libreoj",
        "uoj",
        "OpenThoughts-114k-Code_decontaminated",
        "verifiable_coding_problems_python",
        "DeepCoder",
        "PRIME",
        "codeforces_cots",
        "opencoder",
        "liveincode_generation",
        "ACECode",
        "KodCode",
    ]:
        from . import coder1

        res = coder1.compute_score(
            solution_str,
            ground_truth,
            extra_info=extra_info,
            debug=False,
            format_reward=0.0,
            answer_reward=1.0,
        )
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    elif data_source in [
        "general-reasoner",
        "Multi-subject-RLVR",
        "Math-RLVR",
        "data_linghangyuan",
    ]:
        from . import math_verify

        res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in [
        "musique_0_20000",
        "multihoprag_0_20000",
        "multihoprag_20000_40000",
        "qa_test",
    ]:
        # Multi-hop QA datasets - use docqa evaluation
        from .long_context import docqa

        res = docqa.compute_score(solution_str, ground_truth)
    elif data_source in [
        "references_contains",
        "ref_contains_avg",
    ]:
        # Simple containment average over references present in answer
        from . import references_contains

        res = references_contains.compute_score(solution_str, ground_truth)
    elif data_source in [
        "long_toc_choices_0_20000",
        "long_toc_choices_20000_40000",
        "long_toc_choices_40000_plus",
    ]:
        # Long context table of contents/choice questions - use long evaluation
        from .long_context import long

        res = long.compute_score(solution_str, ground_truth)
    elif data_source in [
        "docmath_0_20000",
        "docmath_20000_40000",
        "docmath_40000_plus",
    ]:
        # Document-based math problems - use docmath evaluation
        from .long_context import docmath

        res = docmath.compute_score(solution_str, ground_truth)
    elif data_source in [
        "passage_count",
    ]:
    
        from .long_context import count
        print(f"[DEBUG] count.compute_score called with data_source: '{data_source}'")
        res = count.compute_score(solution_str, ground_truth)
    elif data_source in [
        "long_context_qa",
        "longcontext_qa",
        "long_context_qa_cn", 
        "qa_direct",
        "math_find",
        "recall",
    ] or data_source.startswith("long_context"):
        # Use unified longqa module for all long-context QA tasks
        from .long_context import longqa
        
        # Set default parameters based on data source
        if data_source == "long_context_qa_cn":
            # Chinese QA with auto-detection
            res = longqa.compute_score(
                solution_str, ground_truth,
                scoring_method="f1",
                language="auto"  # Will detect Chinese automatically
            )
        elif data_source == "recall" or data_source.startswith("long_context_recall"):
            res = longqa.compute_score(
                solution_str, ground_truth,
                scoring_method="recall",
                language="auto"
            )
        elif data_source == "f1" or data_source.startswith("long_context_f1"):
            res = longqa.compute_score(
                solution_str, ground_truth,
                scoring_method="f1",
                language="auto"
            )
        elif data_source.startswith("long_context_llmjudge"):
            res = longqa.compute_score(
                solution_str, ground_truth,
                scoring_method="llm_judge",
                language="auto"
            )
        elif data_source.startswith("long_context_meta"):
            res = longqa.compute_score(
                solution_str, ground_truth,
                scoring_method="metarecall",
                language="auto",
                meta_reward=0.3
            )
        elif data_source.startswith("long_context_orderedlist"):
            res = longqa.compute_score(
                solution_str, ground_truth,
                scoring_method="orderedlist",
                language="auto"
            )
        elif data_source.startswith("long_context_fulltext_recall"):
            res = longqa.compute_score(
                solution_str, ground_truth,
                scoring_method="fulltext_recall",
                language="auto"
            )
        elif data_source == "qa_direct":
            # Direct QA with more pattern matching
            res = longqa.compute_score(
                solution_str, ground_truth,
                scoring_method="f1",  # More flexible scoring
                language="auto"
            )
        elif data_source == "math_find":
            res = longqa.compute_score(
                solution_str, ground_truth,
                scoring_method="contains",
                language="en"
            )
        elif data_source == "ruler":
            res = longqa.compute_score(
                solution_str, ground_truth,
                scoring_method="sub_em",
                language="auto"
            )
        else:  # "long_context_qa" / "longcontext_qa"
            res = longqa.compute_score(
                solution_str, ground_truth,
                scoring_method="f1",
                language="auto"
            )
    elif data_source in [
        "ruler_cwe",
        "ruler_niah",
        "ruler_fwe",
        "ruler_qa",
        "ruler_vt",
        "ruler_other",
    ]:
        from .long_context import ruler
        if data_source in ['ruler_qa']:
            res = ruler.compute_score(solution_str, ground_truth, scoring_method="any")
        else:
            res = ruler.compute_score(solution_str, ground_truth)
    elif data_source in [
        "hotpotqa_evidence",
        "evidence_qa",
    ]:
        from . import evidence_reward
    
        if extra_info is None:
            extra_info = {}
#        if 'question' in (extra_info or {}):
#            extra_info['prompt_str'] = extra_info['question']
    
        res = evidence_reward.default_compute_score(
        data_source, solution_str, ground_truth, extra_info
    )
    else:
        raise NotImplementedError(
            f"Reward function is not implemented for {data_source=}"
        )
    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
    

@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        sandbox_fusion_url,
        concurrent_semaphore,
    )


__all__ = ["default_compute_score"]
