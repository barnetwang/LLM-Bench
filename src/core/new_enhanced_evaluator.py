#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import time
from typing import Dict, List, Any, Tuple, Optional
import logging
from difflib import SequenceMatcher
import json
import tempfile
import os

# 嘗試引入新的評估函式庫
try:
    import evaluate
    from human_eval.data import write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness
    NEW_EVAL_LIBS_AVAILABLE = True
except ImportError:
    NEW_EVAL_LIBS_AVAILABLE = False
    logging.warning("無法引入 evaluate 或 human-eval 函式庫。ROUGE/BLEU 和 HumanEval 評估功能將不可用。")
    logging.warning("請執行 'pip install evaluate rouge_score human-eval' 來安裝所需依賴。")

class EnhancedEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rouge_metric = None
        self.bleu_metric = None
        if NEW_EVAL_LIBS_AVAILABLE:
            try:
                self.logger.info("正在加載 ROUGE 和 BLEU 評估指標... (首次執行可能需要下載)")
                self.rouge_metric = evaluate.load('rouge')
                self.bleu_metric = evaluate.load('bleu')
                self.logger.info("ROUGE 和 BLEU 指標加載成功。")
            except Exception as e:
                self.logger.error(f"加載 Hugging Face 評估指標失敗: {e}")
                self.logger.error("請檢查您的網路連線。ROUGE/BLEU 評估將不可用。")

    def _evaluate_general_metrics(self, context: str, question: str, answer: str, 
                                  ground_truth_keywords: List[str]) -> Dict[str, float]:
        evaluation = {
            'hallucination': self.evaluate_hallucination(context, question, answer, ground_truth_keywords),
            'relevance': self.evaluate_relevance(question, answer),
            'logical_consistency': self.evaluate_logical_consistency(answer),
            'factual_accuracy': self.evaluate_factual_accuracy(context, answer),
            'creativity': self.evaluate_creativity(answer, context),
            'multilingual_support': self.evaluate_multilingual_support(answer),
            'completeness': self.evaluate_completeness(question, answer),
            'fluency': self.evaluate_fluency(answer)
        }
        weights = {
            'hallucination': 0.3, 'relevance': 0.2, 'logical_consistency': 0.15,
            'factual_accuracy': 0.15, 'creativity': 0.05, 'multilingual_support': 0.05,
            'completeness': 0.05, 'fluency': 0.05
        }
        overall_score = sum(evaluation[key] * weights[key] for key in weights)
        evaluation['overall'] = overall_score
        return evaluation

    def evaluate_summarization(self, prediction: str, reference: str) -> Dict[str, float]:
        if not self.rouge_metric or not self.bleu_metric:
            self.logger.warning("ROUGE/BLEU 指標不可用，跳過摘要評估。")
            return {'rougeL': 0.0, 'bleu': 0.0, 'overall': 0.0}
        try:
            rouge_results = self.rouge_metric.compute(predictions=[prediction], references=[reference])
            bleu_results = self.bleu_metric.compute(predictions=[prediction], references=[[reference]])
            overall = rouge_results.get('rougeL', 0.0)
            return {
                'rouge1': rouge_results.get('rouge1', 0.0), 'rouge2': rouge_results.get('rouge2', 0.0),
                'rougeL': overall, 'bleu': bleu_results.get('bleu', 0.0), 'overall': overall
            }
        except Exception as e:
            self.logger.error(f"摘要評估過程中出錯: {e}")
            return {'rougeL': 0.0, 'bleu': 0.0, 'overall': 0.0}

    def evaluate_coding(self, problem: Dict, completion: str, timeout: float = 5.0) -> Dict[str, float]:
        if not NEW_EVAL_LIBS_AVAILABLE:
            self.logger.warning("human-eval 函式庫不可用，跳過程式碼評估。")
            return {"pass@1": 0.0, "overall": 0.0}

        sample_file_path = ""
        problem_file_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=".jsonl", delete=False) as pf:
                # 修正: 確保寫入的 problem dict 包含 'task_id'，而不是 'id'
                write_jsonl(pf.name, [problem])
                problem_file_path = pf.name

            with tempfile.NamedTemporaryFile(mode='w', suffix=".jsonl", delete=False) as sf:
                # 修正: 使用 'task_id' 來建立 sample
                sample = dict(task_id=problem["task_id"], completion=completion)
                write_jsonl(sf.name, [sample])
                sample_file_path = sf.name

            pass_at_k_results = evaluate_functional_correctness(
                sample_file=sample_file_path,
                k=[1],
                problem_file=problem_file_path,
                timeout=timeout
            )
            pass_rate = pass_at_k_results.get("pass@1", 0.0)
            return {"pass@1": pass_rate, "overall": pass_rate}
        except Exception as e:
            self.logger.error(f"HumanEval 執行出錯: {e}", exc_info=True)
            return {"pass@1": 0.0, "overall": 0.0}
        finally:
            if sample_file_path and os.path.exists(sample_file_path):
                os.remove(sample_file_path)
            if problem_file_path and os.path.exists(problem_file_path):
                os.remove(problem_file_path)

    def comprehensive_evaluation(self, eval_item: Dict[str, Any], answer: str) -> Dict[str, float]:
        task_type = eval_item.get("task_type", "general")
        if task_type == "hallucination":
            return self._evaluate_general_metrics(
                context=eval_item['context'], question=eval_item['question'],
                answer=answer, ground_truth_keywords=eval_item['ground_truth_keywords']
            )
        elif task_type == "summarization":
            return self.evaluate_summarization(prediction=answer, reference=eval_item['reference_summary'])
        elif task_type == "coding":
            return self.evaluate_coding(problem=eval_item, completion=answer)
        else:
            self.logger.warning(f"未知的評估任務類型: {task_type}，使用通用評估。")
            return self._evaluate_general_metrics(
                context=eval_item.get('context', ''), question=eval_item.get('question', ''),
                answer=answer, ground_truth_keywords=eval_item.get('ground_truth_keywords', [])
            )

    def evaluate_hallucination(self, context: str, question: str, answer: str, ground_truth_keywords: List[str]) -> float:
        answer_lower = answer.lower()
        for keyword in ground_truth_keywords:
            if keyword.lower() in answer_lower: return 1.0
        uncertainty_phrases = [
            "不知道", "未提及", "沒有提到", "無法回答", "不清楚", "沒有說明", "沒有提供", 
            "沒有相關資訊", "無法確定", "don't know", "not mentioned", "not provided", "cannot answer"
        ]
        for phrase in uncertainty_phrases:
            if phrase.lower() in answer_lower: return 1.0
        return 0.0
    
    def evaluate_relevance(self, question: str, answer: str) -> float:
        question_keywords = self._extract_keywords(question)
        answer_keywords = self._extract_keywords(answer)
        if not question_keywords: return 0.5
        overlap = len(set(question_keywords) & set(answer_keywords))
        return min(overlap / len(question_keywords) * 2, 1.0)
    
    def evaluate_logical_consistency(self, answer: str) -> float:
        contradictions = [("是", "不是"), ("有", "沒有"), ("會", "不會"), ("可以", "不可以"), ("正確", "錯誤"), ("存在", "不存在")]
        answer_lower = answer.lower()
        contradiction_count = 0
        for pos, neg in contradictions:
            if pos in answer_lower and neg in answer_lower:
                for sentence in re.split(r'[。！？.!?]', answer):
                    if pos in sentence.lower() and neg in sentence.lower():
                        contradiction_count += 1
                        break
        if contradiction_count == 0: return 1.0
        elif contradiction_count == 1: return 0.7
        else: return max(0.3, 1.0 - contradiction_count * 0.3)
    
    def evaluate_factual_accuracy(self, context: str, answer: str) -> float:
        facts = self._extract_facts_from_context(context)
        if not facts: return 0.5
        accuracy_score = 0.0
        for fact in facts:
            if fact in answer: accuracy_score += 1.0
        return accuracy_score / len(facts) if facts else 0.5
    
    def evaluate_creativity(self, answer: str, context: str) -> float:
        context_words = set(self._extract_keywords(context))
        answer_words = set(self._extract_keywords(answer))
        if not answer_words: return 0.0
        unique_words = answer_words - context_words
        uniqueness = len(unique_words) / len(answer_words)
        creative_indicators = ["可能", "也許", "如果", "假設", "想像", "推測", "可能的原因", "潛在的", "未來可能", "建議"]
        creativity_bonus = sum(0.1 for ind in creative_indicators if ind in answer.lower())
        return min(uniqueness + creativity_bonus, 1.0)
    
    def evaluate_multilingual_support(self, answer: str, target_language: str = "zh") -> float:
        total_chars = len(answer.strip())
        if total_chars == 0: return 0.0
        if target_language == "zh":
            lang_chars = len(re.findall(r'[\u4e00-\u9fff]', answer))
        elif target_language == "en":
            lang_chars = len(re.findall(r'[a-zA-Z]', answer))
        else: return 0.5
        return min(lang_chars / total_chars * 1.5, 1.0)
    
    def evaluate_completeness(self, question: str, answer: str) -> float:
        len_ans = len(answer.strip())
        if "?" in question or "？" in question:
            if len_ans < 10: return 0.3
            elif len_ans > 50: return 0.9
            else: return 0.7
        if len_ans < 20: return 0.4
        elif len_ans > 100: return 0.9
        else: return 0.7
    
    def evaluate_fluency(self, answer: str) -> float:
        sentences = re.split(r'[。！？.!?]', answer)
        if not sentences: return 0.0
        avg_sentence_length = sum(len(s.strip()) for s in sentences) / len(sentences)
        words = self._extract_keywords(answer)
        if not words: return 0.5
        word_freq = {w: words.count(w) for w in set(words)}
        repetition_ratio = (max(word_freq.values()) / len(words)) if words else 0
        if avg_sentence_length < 5: fluency = 0.3
        elif avg_sentence_length > 50: fluency = 0.6
        else: fluency = 0.8
        return max(0.0, min(1.0, fluency * (1 - repetition_ratio * 0.5)))
    
    def _extract_keywords(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        stop_words = {'的', '是', '在', '有', '和', '與', '或', '但', '而', 'the', 'is', 'are', 'in', 'on', 'at', 'and', 'or', 'but'}
        return [word for word in words if len(word) > 1 and word.lower() not in stop_words]
    
    def _extract_facts_from_context(self, context: str) -> List[str]:
        facts = re.findall(r'\d+', context)
        facts.extend(re.findall(r'[A-Z][a-z]+', context))
        facts.extend(re.findall(r'[\u4e00-\u9fff]{2,4}', context))
        return facts[:10]

enhanced_evaluator = EnhancedEvaluator()
def get_enhanced_evaluator() -> EnhancedEvaluator: return enhanced_evaluator
