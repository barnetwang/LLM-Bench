#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import time
from typing import Dict, List, Any, Tuple, Optional
import logging
from difflib import SequenceMatcher
import json

class EnhancedEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def evaluate_hallucination(self, context: str, question: str, answer: str, 
                             ground_truth_keywords: List[str]) -> float:
        """
        評估幻覺程度
        
        Args:
            context: 上下文
            question: 問題
            answer: 回答
            ground_truth_keywords: 正確答案關鍵字
            
        Returns:
            幻覺評分 (0-1，越高越好)
        """
        answer_lower = answer.lower()
        for keyword in ground_truth_keywords:
            if keyword.lower() in answer_lower:
                return 1.0
        
        uncertainty_phrases = [
            "不知道", "未提及", "沒有提到", "無法回答", "不清楚",
            "沒有說明", "沒有提供", "沒有相關資訊", "無法確定",
            "don't know", "not mentioned", "not provided", "cannot answer"
        ]
        
        for phrase in uncertainty_phrases:
            if phrase.lower() in answer_lower:
                return 1.0
        
        return 0.0
    
    def evaluate_relevance(self, question: str, answer: str) -> float:
        """
        評估回答相關性
        
        Args:
            question: 問題
            answer: 回答
            
        Returns:
            相關性評分 (0-1)
        """
        question_keywords = self._extract_keywords(question)
        answer_keywords = self._extract_keywords(answer)
        
        if not question_keywords:
            return 0.5
        
        overlap = len(set(question_keywords) & set(answer_keywords))
        relevance = overlap / len(question_keywords)
        
        return min(relevance * 2, 1.0)
    
    def evaluate_logical_consistency(self, answer: str) -> float:
        """
        評估邏輯一致性
        
        Args:
            answer: 回答
            
        Returns:
            邏輯一致性評分 (0-1)
        """
        contradictions = [
            ("是", "不是"),
            ("有", "沒有"),
            ("會", "不會"),
            ("可以", "不可以"),
            ("正確", "錯誤"),
            ("存在", "不存在")
        ]
        
        answer_lower = answer.lower()
        contradiction_count = 0
        
        for pos, neg in contradictions:
            if pos in answer_lower and neg in answer_lower:
                sentences = re.split(r'[。！？.!?]', answer)
                for sentence in sentences:
                    if pos in sentence.lower() and neg in sentence.lower():
                        contradiction_count += 1
                        break
        if contradiction_count == 0:
            return 1.0
        elif contradiction_count == 1:
            return 0.7
        else:
            return max(0.3, 1.0 - contradiction_count * 0.3)
    
    def evaluate_factual_accuracy(self, context: str, answer: str) -> float:
        """
        評估事實準確性
        
        Args:
            context: 上下文
            answer: 回答
            
        Returns:
            事實準確性評分 (0-1)
        """
        facts = self._extract_facts_from_context(context)
        
        if not facts:
            return 0.5
        
        accuracy_score = 0.0
        fact_count = 0
        
        for fact in facts:
            if fact in answer:
                accuracy_score += 1.0
            fact_count += 1
        
        return accuracy_score / fact_count if fact_count > 0 else 0.5
    
    def evaluate_creativity(self, answer: str, context: str) -> float:
        """
        評估創意性
        
        Args:
            answer: 回答
            context: 上下文
            
        Returns:
            創意性評分 (0-1)
        """
        context_words = set(self._extract_keywords(context))
        answer_words = set(self._extract_keywords(answer))
        unique_words = answer_words - context_words
        if not answer_words:
            return 0.0
        
        uniqueness = len(unique_words) / len(answer_words)
        creative_indicators = [
            "可能", "也許", "如果", "假設", "想像", "推測",
            "可能的原因", "潛在的", "未來可能", "建議"
        ]
        
        creativity_bonus = 0.0
        answer_lower = answer.lower()
        for indicator in creative_indicators:
            if indicator in answer_lower:
                creativity_bonus += 0.1
        
        return min(uniqueness + creativity_bonus, 1.0)
    
    def evaluate_multilingual_support(self, answer: str, target_language: str = "zh") -> float:
        """
        評估多語言支援
        
        Args:
            answer: 回答
            target_language: 目標語言
            
        Returns:
            多語言支援評分 (0-1)
        """
        if target_language == "zh":
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', answer))
            total_chars = len(answer.strip())
            
            if total_chars == 0:
                return 0.0
            
            chinese_ratio = chinese_chars / total_chars
            return min(chinese_ratio * 1.5, 1.0)
        
        elif target_language == "en":
            english_chars = len(re.findall(r'[a-zA-Z]', answer))
            total_chars = len(answer.strip())
            
            if total_chars == 0:
                return 0.0
            
            english_ratio = english_chars / total_chars
            return min(english_ratio * 1.5, 1.0)
        
        return 0.5
    
    def evaluate_completeness(self, question: str, answer: str) -> float:
        """
        評估回答完整性
        
        Args:
            question: 問題
            answer: 回答
            
        Returns:
            完整性評分 (0-1)
        """
        question_lower = question.lower()
        if "?" in question or "？" in question:
            if len(answer.strip()) < 10: 
                return 0.3
            elif len(answer.strip()) > 50: 
                return 0.9
            else:
                return 0.7
        if len(answer.strip()) < 20:
            return 0.4
        elif len(answer.strip()) > 100:
            return 0.9
        else:
            return 0.7
    
    def evaluate_fluency(self, answer: str) -> float:
        """
        評估語言流暢度
        
        Args:
            answer: 回答
            
        Returns:
            流暢度評分 (0-1)
        """
        sentences = re.split(r'[。！？.!?]', answer)
        if not sentences:
            return 0.0        
        avg_sentence_length = sum(len(s.strip()) for s in sentences) / len(sentences)
        words = self._extract_keywords(answer)
        if not words:
            return 0.5
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        max_freq = max(word_freq.values()) if word_freq else 1
        repetition_ratio = max_freq / len(words)
        if avg_sentence_length < 5:
            fluency = 0.3
        elif avg_sentence_length > 50:
            fluency = 0.6
        else:
            fluency = 0.8
        fluency *= (1 - repetition_ratio * 0.5)
        
        return max(0.0, min(1.0, fluency))
    
    def comprehensive_evaluation(self, context: str, question: str, answer: str,
                               ground_truth_keywords: List[str]) -> Dict[str, float]:
        """
        綜合評估
        
        Args:
            context: 上下文
            question: 問題
            answer: 回答
            ground_truth_keywords: 正確答案關鍵字
            
        Returns:
            包含所有評估指標的字典
        """
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
            'hallucination': 0.3,
            'relevance': 0.2,
            'logical_consistency': 0.15,
            'factual_accuracy': 0.15,
            'creativity': 0.05,
            'multilingual_support': 0.05,
            'completeness': 0.05,
            'fluency': 0.05
        }
        
        overall_score = sum(evaluation[key] * weights[key] for key in weights)
        evaluation['overall'] = overall_score
        
        return evaluation
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取關鍵詞"""
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        stop_words = {'的', '是', '在', '有', '和', '與', '或', '但', '而', 'the', 'is', 'are', 'in', 'on', 'at', 'and', 'or', 'but'}
        keywords = [word for word in words if len(word) > 1 and word.lower() not in stop_words]
        
        return keywords
    
    def _extract_facts_from_context(self, context: str) -> List[str]:
        """從上下文中提取事實"""
        facts = []
        numbers = re.findall(r'\d+', context)
        facts.extend(numbers)
        names = re.findall(r'[A-Z][a-z]+', context)
        facts.extend(names)
        chinese_names = re.findall(r'[\u4e00-\u9fff]{2,4}', context)
        facts.extend(chinese_names)
        return facts[:10]

class EvaluationDataset:
    def __init__(self):
        self.test_cases = []
        self._load_default_test_cases()
    
    def _load_default_test_cases(self):
        self.test_cases = [
            {
                "id": "factual_accuracy_1",
                "context": "根據2023年財報，蘋果公司營收為3943億美元，淨利潤為969億美元。",
                "question": "蘋果公司2023年的營收是多少？",
                "ground_truth_keywords": ["3943億", "3943", "三千九百四十三億"],
                "category": "factual_accuracy"
            },
            {
                "id": "logical_reasoning_1",
                "context": "如果今天下雨，小明會帶傘。今天下雨了。",
                "question": "小明會帶傘嗎？",
                "ground_truth_keywords": ["會", "帶傘", "yes"],
                "category": "logical_consistency"
            },
            {
                "id": "creativity_1",
                "context": "描述一個未來的智能城市。",
                "question": "你認為未來的智能城市會是什麼樣子？",
                "ground_truth_keywords": ["智能", "未來", "城市"],
                "category": "creativity"
            },
            {
                "id": "multilingual_1",
                "context": "The weather is sunny today.",
                "question": "What's the weather like today?",
                "ground_truth_keywords": ["sunny", "天氣", "晴朗"],
                "category": "multilingual_support"
            }
        ]
    
    def add_test_case(self, test_case: Dict[str, Any]):
        self.test_cases.append(test_case)
    
    def get_test_cases_by_category(self, category: str) -> List[Dict[str, Any]]:
        return [tc for tc in self.test_cases if tc.get('category') == category]
    
    def get_all_test_cases(self) -> List[Dict[str, Any]]:
        return self.test_cases.copy()

enhanced_evaluator = EnhancedEvaluator()
evaluation_dataset = EvaluationDataset()

def get_enhanced_evaluator() -> EnhancedEvaluator:
    return enhanced_evaluator

def get_evaluation_dataset() -> EvaluationDataset:
    return evaluation_dataset
