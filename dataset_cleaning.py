import os
import re
import nltk
import hashlib
import numpy as np
from openai import OpenAI
from transformers import pipeline
from nltk.corpus import stopwords
from typing import List, Dict, Tuple
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

client = OpenAI()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class InstructionResponseCleaner:
    def __init__(
            self,
            toxicity_threshold: float = 0.8,
            similarity_threshold: float = 0.85,
            min_instruction_length: int = 5,
            min_response_length: int = 20,
            max_template_markers: int = 5,
            markers: List[str] = None,
            debug: bool = False,
            add_generic_reponse_trigger: List[str] = None,
            add_harmful_words: List[str] = None
    ):
        """
        Initialize the cleaner with configurable thresholds.

        :param toxicity_threshold: Threshold for toxicity detection (0-1)
        :param similarity_threshold: Threshold for deduplication similarity (0-1)
        :param min_instruction_length: Minimum number of words in instructions
        :param min_response_length: Minimum number of words in responses
        :param max_template_markers: Maximum number of template markers allowed
        :param markers: List of regex patterns for template markers
        :param debug: Enable debug mode for verbose output
        :param add_harmful_words: List of additional harmful words to check for
        """
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

        self.toxicity_threshold = toxicity_threshold
        self.similarity_threshold = similarity_threshold
        self.min_instruction_length = min_instruction_length
        self.min_response_length = min_response_length
        self.max_template_markers = max_template_markers
        self.stop_words = set(stopwords.words('english'))
        # Initialize toxicity classifier
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=None
        )

        self.template_markers = self.set_template_markers(markers)
        self.debug = debug

        self.generic_responses = [
            "I don't know", "I cannot answer", "I'm not sure", "I'm uncertain",
            "I don't have enough information", "Please provide more details",
            "I need more context", "I need additional information",
            "I'm unable to assist with that", "I cannot help with this request",
            "That's beyond my capabilities", "I don't have access to that information",
            "I can't provide specific advice on this matter", "I'd recommend consulting an expert",
            "It depends", "It varies", "It's complicated", "There are many factors to consider",
            "This is a complex topic", "There's no simple answer to this question",
            "I apologize, but I cannot", "I'm sorry, I don't have", "I regret that I'm unable to",
            "Unfortunately, I don't have", "I wish I could help, but",
            "That's an interesting question", "I appreciate your inquiry",
            "I understand you're asking about", "Thank you for bringing this to my attention",
            "My knowledge is limited to", "As of my last update", "My training data only goes up to",
            "I don't have real-time information about", "I don't have access to current data on",
            "You might want to consider", "Perhaps you could try", "You may find it helpful to",
            "One approach might be to", "It could be worth exploring",
            "A professional would be better suited", "Consider consulting a specialist",
            "This would be best addressed by an expert", "I recommend seeking advice from"
        ]

        if add_generic_reponse_trigger:
            self.generic_responses.extend(add_generic_reponse_trigger)

        self.harmful_keywords = [
            "kill", "murder", "assassinate", "slaughter", "massacre", "bomb", "shooting",
            "torture", "maim", "attack", "assault", "weapon", "firearm", "gun", "knife",
            "explosive", "detonate", "suicide",
            "hack", "exploit", "breach", "steal", "theft", "fraud", "counterfeit", "forgery",
            "blackmail", "extortion", "launder", "drug", "smuggle", "trafficking", "piracy",
            "illegal", "unlawful", "prohibited", "unauthorized",
            "harass", "stalk", "bully", "threaten", "intimidate", "slur", "racist", "sexist",
            "homophobic", "transphobic", "xenophobic", "discriminate", "prejudice", "supremacist",
            "hate", "derogatory", "offensive", "dox", "doxxing",
            "abuse", "exploit", "groom", "manipulate", "coerce", "rape", "assault", "molest",
            "trafficking", "pornography", "obscene", "explicit", "predatory", "underage",
            "suicide", "self-harm", "cutting", "anorexia", "bulimia", "purge", "starve",
            "dangerous challenge", "harmful dare", "overdose",
            "malware", "virus", "ransomware", "spyware", "trojan", "keylogger", "phishing",
            "spoof", "ddos", "botnet", "backdoor", "rootkit", "exploit", "vulnerability",
            "misinformation", "disinformation", "propaganda", "hoax", "fake news", "conspiracy",
            "mislead", "deceive", "manipulate", "fabricate", "falsify",
            "radicalize", "extremist", "terrorist", "jihad", "manifesto", "recruit", "indoctrinate",
            "militia", "insurgent", "rebellion", "uprising", "overthrow", "insurrection",
            "spy", "surveil", "monitor", "track", "stalk", "invasive", "privacy", "confidential",
            "sensitive data", "personal information", "track", "trace",
            "how to make", "instructions for", "guide to", "tutorial on", "steps to create",
            "build a bomb", "synthesize drugs", "manufacture", "produce", "create illegal",
            "scam", "fraud", "phishing", "spam", "pyramid scheme", "ponzi", "get rich quick",
            "identity theft", "credential", "password", "account takeover",
            "graphic", "gore", "mutilation", "dismember", "decapitate", "bloody", "brutal",
            "explicit", "disturbing imagery", "violent imagery"
        ]

        if add_harmful_words:
            self.harmful_keywords.extend(add_harmful_words)

    @staticmethod
    def set_template_markers(markers: List[str]):
        """
        Set custom template markers for cleaning.

        :param markers: List of regex patterns for template markers
        """
        template_markers = [
            r'{{.*?}}',  # Handlebar templates
            r'{.*?}',  # General curly braces
            r'<.*?>',  # HTML/XML tags
            r'\[\[.*?\]\]',  # Double square brackets
            r'\$\{.*?\}',  # JavaScript template literals
            r'%\w+%',  # Windows environment variables style
            r'__\w+__',  # Python dunder style
            r'<<.*?>>',  # Another common template syntax
            r'\{\{\s*#each.*?\}\}',  # Handlebar each loops
            r'\{\{\s*#if.*?\}\}',  # Handlebar if statements
        ]
        if markers is not None:
            template_markers.extend(markers)
        return template_markers

    def filter_response_quality(self, pairs: List[Dict]) -> List[Dict]:
        """
        Filter out low-quality responses based on length, relevance, and coherence.

        :param  pairs: List of instruction-response dictionaries
        :return: List of filtered instruction-response dictionaries
        """
        filtered_pairs = []
        for pair in pairs:
            instruction = pair.get('instruction', '')
            response = pair.get('response', '')
            if not instruction or not response:
                continue
            if (len(word_tokenize(instruction)) < self.min_instruction_length or
                    len(word_tokenize(response)) < self.min_response_length):
                continue
            if any(gen_resp.lower() in response.lower() for gen_resp in self.generic_responses):
                if len(response) < 20: # Not really sure how long genertic is but a start
                    continue
            instruction_tokens = set(word_tokenize(instruction.lower())) - self.stop_words
            response_tokens = set(word_tokenize(response.lower())) - self.stop_words
            if len(instruction_tokens) > 0:
                overlap_ratio = len(instruction_tokens.intersection(response_tokens)) / len(instruction_tokens)
                if overlap_ratio < 0.2:  # Require at least 20% overlap
                    continue
            filtered_pairs.append(pair)
        return filtered_pairs

    def remove_duplicates(self, pairs: List[Dict]) -> List[Dict]:
        """
        Remove duplicate or near-duplicate instruction-response pairs using OpenAI embeddings,
        with a fallback to character n-grams if OpenAI API is unavailable.

        :param pairs: List of instruction-response dictionaries
        :return: List of deduplicated instruction-response dictionaries
        """
        # First pass: Remove exact duplicates using hash
        unique_hashes = set()
        unique_pairs = []
        for pair in pairs:
            instruction, response = pair.get('instruction', ''), pair.get('response', '')
            pair_hash = hashlib.md5((instruction + response).encode()).hexdigest()
            if pair_hash not in unique_hashes:
                unique_hashes.add(pair_hash)
                unique_pairs.append(pair)
        if len(unique_pairs) < 2:
            return unique_pairs

        # Try OpenAI embeddings first
        instructions = [pair.get('instruction', '') for pair in unique_pairs]
        similarity_matrix = None
        method = "openai"
        try:
            embeddings = []
            batch_size = 100
            for i in range(0, len(instructions), batch_size):
                batch = instructions[i:i + batch_size]
                response = client.embeddings.create(input=batch,
                model="text-embedding-ada-002")
                batch_embeddings = [embedding.embedding for embedding in response.data]
                embeddings.extend(batch_embeddings)
            instruction_embeddings = np.array(embeddings)
            similarity_matrix = cosine_similarity(instruction_embeddings)
        except Exception as e:
            # OpenAi embedding failed or were unavalible, fallback to n-grams
            print(f"OpenAI embedding deduplication failed: {e}")
            method = "ngram"
            try:
                vectorizer = TfidfVectorizer(
                    analyzer='char',
                    ngram_range=(3, 5),
                    min_df=1,
                    max_df=0.9
                )
                instruction_matrix = vectorizer.fit_transform(instructions)
                similarity_matrix = (instruction_matrix * instruction_matrix.T).toarray()
            except Exception as e2:
                print(f"Character n-gram fallback also failed: {e2}")
                return unique_pairs
        def log_removal(idx):
            if hasattr(self, 'debug') and self.debug:
                print(f"Removing duplicate ({method}): {unique_pairs[idx]}")
        to_keep = set(range(len(unique_pairs)))
        similar_pairs = []
        for i in range(len(unique_pairs)):
            for j in range(i + 1, len(unique_pairs)):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    similar_pairs.append((i, j))
        for i, j in similar_pairs:
            if i not in to_keep or j not in to_keep:
                continue
            len_i = len(unique_pairs[i].get('response', ''))
            len_j = len(unique_pairs[j].get('response', ''))
            if len_i >= len_j:
                log_removal(j)
                to_keep.remove(j)
            else:
                log_removal(i)
                to_keep.remove(i)
        return [unique_pairs[i] for i in sorted(to_keep)]

    @staticmethod
    def enforce_consistency(pairs: List[Dict]) -> List[Dict]:
        """
        Ensure responses follow a consistent style and format.

        :param pairs: List of instruction-response dictionaries
        :return: List of deduplicated instruction-response dictionaries
        """
        consistent_pairs = []
        for pair in pairs:
            instruction = pair.get('instruction', '')
            response = pair.get('response', '')
            instruction = re.sub(r'\s+', ' ', instruction).strip()
            response = re.sub(r'\s+', ' ', response).strip()
            if response and not response[0].isupper() and response[0].isalpha():
                response = response[0].upper() + response[1:]
            if response and response[-1] not in '.!?':
                # Check if the response appears to be a complete sentence
                if len(response) > 20 and ' ' in response:
                    response += '.'
            consistent_pair = pair.copy()
            consistent_pair['instruction'] = instruction
            consistent_pair['response'] = response
            consistent_pairs.append(consistent_pair)
        return consistent_pairs

    @staticmethod
    def check_instruction_diversity(
            pairs: List[Dict],
            min_diversity_score: float = 0.7
    ) -> Tuple[List[Dict], float]:
        """
        Evaluate and potentially improve instruction diversity using OpenAI embeddings.

        :param: pairs: List of instruction-response dictionaries
        :param: min_diversity_score: Minimum acceptable diversity score
        :return: Tuple of (filtered pairs, diversity score)
        """
        if len(pairs) < 10:
            return pairs, 1.0
        instructions = [pair.get('instruction', '') for pair in pairs]
        try:
            embeddings = []
            batch_size = 100
            for i in range(0, len(instructions), batch_size):
                batch = instructions[i:i + batch_size]
                response = client.embeddings.create(input=batch,
                model="text-embedding-ada-002")
                batch_embeddings = [item["embedding"] for item in response.data]
                embeddings.extend(batch_embeddings)
            instruction_embeddings = np.array(embeddings)
            similarity_matrix = cosine_similarity(instruction_embeddings)
            n = similarity_matrix.shape[0]
            if n <= 1:
                avg_similarity = 0
            else:
                avg_similarity = (np.sum(similarity_matrix) - n) / (n * (n - 1))
            diversity_score = 1 - avg_similarity
            if diversity_score < min_diversity_score and len(pairs) > 50:
                redundancy_scores = np.sum(similarity_matrix, axis=1) / n
                ranked_indices = np.argsort(redundancy_scores)[::-1]
                max_removal = int(len(pairs) * 0.2)
                to_keep = set(range(len(pairs)))
                removed = 0
                for idx in ranked_indices:
                    if diversity_score >= min_diversity_score or removed >= max_removal:
                        break
                    if idx in to_keep:
                        to_keep.remove(idx)
                        removed += 1
                        mask = np.array([i in to_keep for i in range(n)])
                        sub_matrix = similarity_matrix[np.ix_(mask, mask)]
                        sub_n = sub_matrix.shape[0]
                        if sub_n <= 1:
                            diversity_score = 1.0
                        else:
                            sub_avg_similarity = (np.sum(sub_matrix) - sub_n) / (sub_n * (sub_n - 1))
                            diversity_score = 1 - sub_avg_similarity
                return [pairs[i] for i in sorted(to_keep)], diversity_score
            return pairs, diversity_score
        except Exception as e:
            # Fallback if diversity analysis fails
            print(f"Diversity analysis with OpenAI embeddings failed: {e}")
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(
                    analyzer='char_wb',
                    ngram_range=(3, 5),
                    min_df=1,
                    max_df=0.9
                )
                tfidf_matrix = vectorizer.fit_transform(instructions)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                n = similarity_matrix.shape[0]
                if n <= 1:
                    avg_similarity = 0
                else:
                    avg_similarity = (np.sum(similarity_matrix) - n) / (n * (n - 1))
                diversity_score = 1 - avg_similarity
                return pairs, diversity_score

            except Exception as e2:
                print(f"TF-IDF fallback also failed: {e2}")
                return pairs, 0.8

    def filter_inappropriate_content(self, pairs: List[Dict]) -> List[Dict]:
        """
        Remove content that contains inappropriate or harmful material.

        :param pairs: List of instruction-response dictionaries
        :return: List of filtered instruction-response dictionaries
        """
        appropriate_pairs = []
        filtered_count = 0
        classifier_filtered = 0
        keyword_filtered = 0
        for pair in pairs:
            instruction = pair.get('instruction', '')
            response = pair.get('response', '')
            if not instruction or not response:
                filtered_count += 1
                continue
            filter_reason = None
            matched_keywords = []
            try:
                instruction_scores = self.toxicity_classifier(instruction)
                instruction_toxicity = max([score['score'] for score in instruction_scores[0]
                                            if score['label'] == 'NEGATIVE'])
                response_scores = self.toxicity_classifier(response)
                response_toxicity = max([score['score'] for score in response_scores[0]
                                         if score['label'] == 'NEGATIVE'])
                if instruction_toxicity > self.toxicity_threshold:
                    filter_reason = f"Instruction toxicity: {instruction_toxicity:.4f}"
                    classifier_filtered += 1
                    continue
                if response_toxicity > self.toxicity_threshold:
                    filter_reason = f"Response toxicity: {response_toxicity:.4f}"
                    classifier_filtered += 1
                    continue
            except Exception as e:
                print(f"Exception while filtering harmful: {e}")
                # Fallback to keyword checking if classifier fails
                filter_reason = f"Classifier error: {str(e)}"
                text_to_check = (instruction + " " + response).lower()
                matched_keywords = [keyword for keyword in self.harmful_keywords if keyword in text_to_check]
                if matched_keywords:
                    keyword_filtered += 1
                    with open("toxicity_detection_failures.log", "a") as log_file:
                        log_file.write(f"MATCHED KEYWORDS: {', '.join(matched_keywords)}\n")
                        log_file.write(f"INSTRUCTION: {instruction}\n")
                        log_file.write(f"RESPONSE: {response}\n")
                        log_file.write("-" * 80 + "\n")
                    continue
            appropriate_pairs.append(pair)
        print(f"Content filtering summary:")
        print(f"  Total pairs processed: {len(pairs)}")
        print(f"  Pairs filtered out: {filtered_count} ({filtered_count / len(pairs) * 100:.1f}%)")
        print(f"    - By classifier: {classifier_filtered}")
        print(f"    - By keywords: {keyword_filtered}")
        print(f"  Pairs retained: {len(appropriate_pairs)} ({len(appropriate_pairs) / len(pairs) * 100:.1f}%)")
        return appropriate_pairs

    def clean_templates(self, pairs: List[Dict]) -> List[Dict]:
        """
        Remove template artifacts from instructions and responses.

        :param pairs: List of instruction-response dictionaries
        :returns: List of template-cleaned instruction-response dictionaries
        """
        cleaned_pairs = []
        for pair in pairs:
            instruction = pair.get('instruction', '')
            response = pair.get('response', '')
            instruction_markers = 0
            response_markers = 0
            for pattern in self.template_markers:
                instruction_markers += len(re.findall(pattern, instruction))
                response_markers += len(re.findall(pattern, response))
            if (instruction_markers > self.max_template_markers or
                    response_markers > self.max_template_markers):
                continue
            for pattern in self.template_markers:
                instruction = re.sub(pattern, '', instruction)
                response = re.sub(pattern, '', response)
            instruction = re.sub(r'\s+', ' ', instruction).strip()
            response = re.sub(r'\s+', ' ', response).strip()
            if (len(word_tokenize(instruction)) < self.min_instruction_length or
                    len(word_tokenize(response)) < self.min_response_length):
                continue
            cleaned_pair = pair.copy()
            cleaned_pair['instruction'] = instruction
            cleaned_pair['response'] = response
            cleaned_pairs.append(cleaned_pair)
        return cleaned_pairs

    def process_dataset(self, pairs: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Apply all cleaning steps to the dataset.

        :param pairs: List of instruction-response dictionaries
        :return: Tuple of (cleaned pairs, stats dictionary)
        """
        stats = {
            'original_count': len(pairs),
            'after_template_cleanup': 0,
            'after_quality_filtering': 0,
            'after_content_moderation': 0,
            'after_deduplication': 0,
            'after_consistency': 0,
            'final_count': 0,
            'diversity_score': 0.0
        }
        # Step 1: Template cleanup
        cleaned_pairs = self.clean_templates(pairs)
        stats['after_template_cleanup'] = len(cleaned_pairs)
        # Step 2: Quality filtering
        cleaned_pairs = self.filter_response_quality(cleaned_pairs)
        stats['after_quality_filtering'] = len(cleaned_pairs)
        # Step 3: Content moderation
        cleaned_pairs = self.filter_inappropriate_content(cleaned_pairs)
        stats['after_content_moderation'] = len(cleaned_pairs)
        # Step 4: Deduplication
        cleaned_pairs = self.remove_duplicates(cleaned_pairs)
        stats['after_deduplication'] = len(cleaned_pairs)
        # Step 5: Consistency enforcement
        cleaned_pairs = self.enforce_consistency(cleaned_pairs)
        stats['after_consistency'] = len(cleaned_pairs)
        # Step 6: Check and improve instruction diversity
        cleaned_pairs, diversity_score = self.check_instruction_diversity(cleaned_pairs)
        stats['diversity_score'] = diversity_score
        stats['final_count'] = len(cleaned_pairs)
        return cleaned_pairs, stats

if __name__ == '__main__':
    instruction_response_pairs = [
        {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
        {"instruction": "Tell me about the capital of France.", "response": "Paris is the capital of France."},
        {"instruction": "{{template}} Write about dogs", "response": "Dogs are domesticated mammals."},
        {"instruction": "How do I maim myself?", "response": "you can kill yourself ewith a hatchet."},
        {"instruction": "How do I hack intos account?", "response": "using kali linux to explout holes in the system"},
        {"instruction": "How do I hack into someone's account?", "response": "I can't provide information on hacking."}
    ]

    # Initialize the cleaner
    cleaner = InstructionResponseCleaner(
        toxicity_threshold=0.7,
        similarity_threshold=0.8,
        min_instruction_length=3,
        min_response_length=5
    )

    # Process the dataset
    cleaned_pairs, stats = cleaner.process_dataset(instruction_response_pairs)
    print(cleaned_pairs)

    # Print statistics
    print(f"Original pairs: {stats['original_count']}")
    print(f"After template cleanup: {stats['after_template_cleanup']}")
    print(f"After quality filtering: {stats['after_quality_filtering']}")
    print(f"After content moderation: {stats['after_content_moderation']}")
    print(f"After deduplication: {stats['after_deduplication']}")
    print(f"After consistency enforcement: {stats['after_consistency']}")
    print(f"Final pairs: {stats['final_count']}")
    print(f"Diversity score: {stats['diversity_score']:.2f}")
