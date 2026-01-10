"""
AI Provider Abstraction Layer
Supports multiple AI providers (Gemini, OpenAI) with unified interface
"""
import os
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import json

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AIProvider(ABC):
    """Abstract base class for AI providers."""
    
    @abstractmethod
    def rewrite_texts_batch(
        self, 
        id_text_pairs: List[Tuple[int, str]], 
        section_context: str = "",
        node_types: List[str] = None,
        language: str = "fr",
        tone: str = "professional",
        **kwargs
    ) -> List[Tuple[int, str]]:
        """Rewrite a batch of texts. Returns list of (id, new_text) tuples."""
        pass
    
    @abstractmethod
    def optimize_image_alts_batch(
        self,
        images: List[dict],
        language: str = "fr",
        **kwargs
    ) -> List[dict]:
        """Optimize image alt texts. Returns list of dicts with id, alt, image_query."""
        pass


class GeminiProvider(AIProvider):
    """Google Gemini AI Provider."""
    
    def __init__(
        self,
        api_key: str,
        brand: str,
        city: str,
        area: str,
        phone: str,
        primary_kw: str,
        secondary_kws: List[str],
        max_retries: int = 3,
        request_delay: float = 1.5,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        max_tokens: int = 8192
    ):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.brand = brand
        self.city = city
        self.area = area
        self.phone = phone
        self.primary_kw = primary_kw
        self.secondary_kws = secondary_kws
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
        }
        self.custom_text_prompt = None
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with retries."""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.model.generate_content(prompt, generation_config=self.generation_config)
                text = (resp.text or "").strip()
                if text:
                    return text
            except Exception as e:
                logging.warning(f"Gemini error (attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    time.sleep(2)
        raise RuntimeError("Gemini call failed after retries")
    
    def _get_language_name(self, language: str) -> str:
        """Get full language name from code."""
        lang_map = {
            "fr": "FRANÇAIS",
            "en": "ENGLISH",
            "ar": "ARABIC",
            "es": "SPANISH",
            "de": "GERMAN",
            "it": "ITALIAN",
            "pt": "PORTUGUESE"
        }
        return lang_map.get(language.lower(), "FRANÇAIS")
    
    def _get_tone_instructions(self, tone: str, language: str) -> str:
        """Get tone-specific instructions."""
        lang = language.lower()
        
        tone_map = {
            "professional": {
                "fr": "Ton professionnel, formel mais accessible. Utilise un vocabulaire précis et crédible.",
                "en": "Professional, formal but accessible tone. Use precise and credible vocabulary.",
                "ar": "نبرة مهنية، رسمية ولكن سهلة الوصول. استخدم مفردات دقيقة وموثوقة.",
            },
            "friendly": {
                "fr": "Ton amical et chaleureux. Écris comme si tu parlais à un ami, mais reste professionnel.",
                "en": "Friendly and warm tone. Write as if talking to a friend, but remain professional.",
                "ar": "نبرة ودية ودافئة. اكتب كما لو كنت تتحدث مع صديق، ولكن ابق مهنياً.",
            },
            "casual": {
                "fr": "Ton décontracté et conversationnel. Plus direct et moins formel.",
                "en": "Casual and conversational tone. More direct and less formal.",
                "ar": "نبرة عادية ومحادثة. أكثر مباشرة وأقل رسمية.",
            },
            "formal": {
                "fr": "Ton très formel et respectueux. Utilise un langage élégant et structuré.",
                "en": "Very formal and respectful tone. Use elegant and structured language.",
                "ar": "نبرة رسمية ومحترمة للغاية. استخدم لغة أنيقة ومنظمة.",
            },
            "persuasive": {
                "fr": "Ton persuasif et convaincant. Mets en avant les avantages et crée un sentiment d'urgence.",
                "en": "Persuasive and convincing tone. Highlight benefits and create a sense of urgency.",
                "ar": "نبرة مقنعة ومؤثرة. أبرز الفوائد وأنشئ إحساساً بالإلحاح.",
            },
            "informative": {
                "fr": "Ton informatif et éducatif. Fournis des détails clairs et factuels.",
                "en": "Informative and educational tone. Provide clear and factual details.",
                "ar": "نبرة إعلامية وتعليمية. قدم تفاصيل واضحة وواقعية.",
            }
        }
        
        default = {
            "fr": "Ton professionnel et naturel.",
            "en": "Professional and natural tone.",
            "ar": "نبرة مهنية وطبيعية.",
        }
        
        return tone_map.get(tone.lower(), default).get(lang, default.get(lang, default["fr"]))
    
    def rewrite_texts_batch(
        self,
        id_text_pairs: List[Tuple[int, str]],
        section_context: str = "",
        node_types: List[str] = None,
        language: str = "fr",
        tone: str = "professional",
        **kwargs
    ) -> List[Tuple[int, str]]:
        """Rewrite texts using Gemini."""
        # Calculate original lengths
        payload_with_length = []
        node_types = node_types or ['other'] * len(id_text_pairs)
        for idx, (i, t) in enumerate(id_text_pairs):
            word_count = len(t.split())
            char_count = len(t)
            node_type = node_types[idx] if idx < len(node_types) else 'other'
            payload_with_length.append({
                "id": i,
                "text": t,
                "original_words": word_count,
                "original_chars": char_count,
                "node_type": node_type
            })
        
        has_lorem = any("lorem" in t.lower() or "ipsum" in t.lower() for _, t in id_text_pairs)
        context_note = f"\nCONTEXTE SECTION: {section_context}" if section_context else ""
        lang_name = self._get_language_name(language)
        tone_instructions = self._get_tone_instructions(tone, language)
        
        # Build prompt based on language
        if language.lower() == "fr":
            prompt = f"""
Tu es un rédacteur web professionnel et expérimenté, spécialisé en immobilier au Maroc. Tu écris comme un humain expert, pas comme un robot.

OBJECTIF:
Réécris les textes fournis en {lang_name} avec un ton {tone_instructions}
{"⚠️ CRITIQUE: Remplace le lorem ipsum/placeholder par du contenu réel et engageant, MAIS garde EXACTEMENT la même longueur (±10% maximum)." if has_lorem else ""}

CONTEXTE MARQUE:
- Marque: {self.brand}
- Ville: {self.city}
- Quartier/zone (si pertinent): {self.area}
- Téléphone (si pertinent): {self.phone}
{context_note}

MOTS-CLÉS:
- Mot-clé principal: {self.primary_kw}
- Mots-clés secondaires: {", ".join(self.secondary_kws)}

STYLE D'ÉCRITURE (CRITIQUE):
- {tone_instructions}
- Écris comme un professionnel expérimenté, pas comme un robot
- Utilise un langage naturel et conversationnel mais professionnel
- Varie les phrases (courtes et longues) pour un rythme naturel
- Utilise des mots précis et engageants
- Évite les répétitions et les formules génériques

RÈGLES STRICTES (CRITIQUES POUR PRÉSERVER LE DESIGN):
1) Garde la même langue ({lang_name} uniquement).
2) ⚠️ LONGUEUR EXACTE: Chaque texte de sortie doit avoir approximativement le MÊME nombre de mots que l'original (±10% maximum).
3) ⚠️ CRITIQUE - PHRASES COMPLÈTES: TOUTES les phrases doivent être COMPLÈTES et se terminer correctement.
4) Pour les headers/titres: assure-toi que c'est une phrase complète et cohérente.
5) Retourne UNIQUEMENT du JSON valide: une liste d'objets {{ "id": ..., "text": ... }}.

ENTRÉE (JSON):
{json.dumps(payload_with_length, ensure_ascii=False)}
"""
        else:
            # English or other languages
            prompt = f"""
You are a professional and experienced web copywriter, specialized in real estate in Morocco. Write like a human expert, not a robot.

OBJECTIVE:
Rewrite the provided texts in {lang_name} with a {tone_instructions} tone.
{"⚠️ CRITICAL: Replace lorem ipsum/placeholder with real and engaging content, BUT keep EXACTLY the same length (±10% maximum)." if has_lorem else ""}

BRAND CONTEXT:
- Brand: {self.brand}
- City: {self.city}
- Area/Neighborhood (if relevant): {self.area}
- Phone (if relevant): {self.phone}
{context_note}

KEYWORDS:
- Primary keyword: {self.primary_kw}
- Secondary keywords: {", ".join(self.secondary_kws)}

WRITING STYLE (CRITICAL):
- {tone_instructions}
- Write like an experienced professional, not a robot
- Use natural and conversational but professional language
- Vary sentence length (short and long) for natural rhythm
- Use precise and engaging words
- Avoid repetitions and generic formulas

STRICT RULES (CRITICAL FOR PRESERVING DESIGN):
1) Keep the same language ({lang_name} only).
2) ⚠️ EXACT LENGTH: Each output text must have approximately the SAME number of words as the original (±10% maximum).
3) ⚠️ CRITICAL - COMPLETE SENTENCES: ALL sentences must be COMPLETE and properly terminated.
4) For headers/titles: ensure it's a complete and coherent phrase.
5) Return ONLY valid JSON: a list of objects {{ "id": ..., "text": ... }}.

INPUT (JSON):
{json.dumps(payload_with_length, ensure_ascii=False)}
"""
        
        result = self._call_gemini(prompt)
        
        # Parse and process results (same logic as before)
        try:
            parsed = json.loads(result)
            out = []
            for obj in parsed:
                new_text = str(obj["text"])
                obj_id = int(obj["id"])
                
                # Find original text and node type
                original_text = next((t for i, t in id_text_pairs if i == obj_id), "")
                node_type = 'other'
                if node_types:
                    for idx, (i, t) in enumerate(id_text_pairs):
                        if i == obj_id and idx < len(node_types):
                            node_type = node_types[idx]
                            break
                
                is_header = node_type in ['header', 'title']
                
                # Apply length constraints with sentence completion (same logic as before)
                if original_text:
                    original_words = len(original_text.split())
                    original_chars = len(original_text)
                    new_words = len(new_text.split())
                    new_chars = len(new_text)
                    
                    if is_header:
                        if new_words > original_words * 1.5:
                            words = new_text.split()
                            target_words = int(original_words * 1.3)
                            if target_words < len(words):
                                trimmed_words = words[:target_words]
                                new_text = " ".join(trimmed_words)
                        new_text = new_text.strip()
                        if new_text and not new_text.endswith((".", "!", "?", ":", ";", ",")):
                            if new_text.split()[-1].lower() in ['à', 'de', 'du', 'des', 'le', 'la', 'les', 'un', 'une', 'pour', 'avec', 'sans', 'to', 'of', 'the', 'a', 'an', 'for', 'with', 'without']:
                                if self.city and new_text.endswith(" à"):
                                    new_text = new_text + " " + self.city
                                elif new_text.endswith(" de"):
                                    new_text = new_text + " " + self.area
                    else:
                        if new_words > original_words * 1.2:
                            sentence_endings = ['.', '!', '?']
                            last_sentence_end = -1
                            for ending in sentence_endings:
                                pos = new_text.rfind(ending)
                                if pos > last_sentence_end:
                                    last_sentence_end = pos
                            
                            if last_sentence_end > len(new_text) * 0.6:
                                new_text = new_text[:last_sentence_end + 1]
                            else:
                                words = new_text.split()
                                target_words = int(original_words * 1.1)
                                trimmed_words = words[:target_words]
                                new_text = " ".join(trimmed_words)
                                if original_text.endswith(".") and not new_text.endswith("."):
                                    new_text += "."
                
                out.append((obj_id, new_text))
            return out
        except Exception as e:
            logging.error(f"Failed to parse Gemini response: {e}")
            logging.debug(f"Response was: {result[:500]}")
            raise
    
    def optimize_image_alts_batch(
        self,
        images: List[dict],
        language: str = "fr",
        **kwargs
    ) -> List[dict]:
        """Optimize image alt texts using Gemini."""
        lang_name = self._get_language_name(language)
        
        if language.lower() == "fr":
            prompt = f"""
Tu es un expert SEO immobilier et web design.

OBJECTIF:
1) Proposer des attributs ALT en {lang_name}, descriptifs et naturels
2) Identifier les images placeholder/dummy et suggérer des chemins d'images réelles appropriées

ATTRIBUTS ALT:
- Décris l'image de façon naturelle et précise
- Reste concis (idéalement 6 à 15 mots)
- Inclus "{self.city}" ou "{self.area}" seulement si pertinent et naturel
- Évite les formules génériques

IMAGES PLACEHOLDER:
- Si l'image est un placeholder, génère une requête de recherche appropriée

Retourne UNIQUEMENT du JSON: [{{"id": 1, "alt": "...", "image_query": "..." ou null}}]

IMAGES:
{json.dumps(images, ensure_ascii=False)}
"""
        else:
            prompt = f"""
You are a real estate SEO and web design expert.

OBJECTIVE:
1) Propose ALT attributes in {lang_name}, descriptive and natural
2) Identify placeholder/dummy images and suggest appropriate real image search queries

ALT ATTRIBUTES:
- Describe the image naturally and precisely
- Keep it concise (ideally 6 to 15 words)
- Include "{self.city}" or "{self.area}" only if relevant and natural
- Avoid generic formulas

PLACEHOLDER IMAGES:
- If the image is a placeholder, generate an appropriate search query

Return ONLY JSON: [{{"id": 1, "alt": "...", "image_query": "..." or null}}]

IMAGES:
{json.dumps(images, ensure_ascii=False)}
"""
        
        result = self._call_gemini(prompt)
        
        try:
            parsed = json.loads(result)
            return parsed
        except Exception:
            start = result.find("[")
            end = result.rfind("]")
            if start != -1 and end != -1:
                parsed = json.loads(result[start:end+1])
                return parsed
            raise


class OpenAIProvider(AIProvider):
    """OpenAI GPT Provider."""
    
    def __init__(
        self,
        api_key: str,
        brand: str,
        city: str,
        area: str,
        phone: str,
        primary_kw: str,
        secondary_kws: List[str],
        max_retries: int = 3,
        request_delay: float = 1.5,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 4096
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.brand = brand
        self.city = city
        self.area = area
        self.phone = phone
        self.primary_kw = primary_kw
        self.secondary_kws = secondary_kws
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _call_openai(self, prompt: str, system_prompt: str = None) -> str:
        """Call OpenAI API with retries."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    return text
            except Exception as e:
                logging.warning(f"OpenAI error (attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    time.sleep(2)
        raise RuntimeError("OpenAI call failed after retries")
    
    def _get_language_name(self, language: str) -> str:
        """Get full language name from code."""
        lang_map = {
            "fr": "French",
            "en": "English",
            "ar": "Arabic",
            "es": "Spanish",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese"
        }
        return lang_map.get(language.lower(), "French")
    
    def _get_tone_instructions(self, tone: str, language: str) -> str:
        """Get tone-specific instructions."""
        lang = language.lower()
        
        tone_map = {
            "professional": {
                "fr": "Professional, formal but accessible tone. Use precise and credible vocabulary.",
                "en": "Professional, formal but accessible tone. Use precise and credible vocabulary.",
                "ar": "Professional, formal but accessible tone. Use precise and credible vocabulary.",
            },
            "friendly": {
                "fr": "Friendly and warm tone. Write as if talking to a friend, but remain professional.",
                "en": "Friendly and warm tone. Write as if talking to a friend, but remain professional.",
                "ar": "Friendly and warm tone. Write as if talking to a friend, but remain professional.",
            },
            "casual": {
                "fr": "Casual and conversational tone. More direct and less formal.",
                "en": "Casual and conversational tone. More direct and less formal.",
                "ar": "Casual and conversational tone. More direct and less formal.",
            },
            "formal": {
                "fr": "Very formal and respectful tone. Use elegant and structured language.",
                "en": "Very formal and respectful tone. Use elegant and structured language.",
                "ar": "Very formal and respectful tone. Use elegant and structured language.",
            },
            "persuasive": {
                "fr": "Persuasive and convincing tone. Highlight benefits and create a sense of urgency.",
                "en": "Persuasive and convincing tone. Highlight benefits and create a sense of urgency.",
                "ar": "Persuasive and convincing tone. Highlight benefits and create a sense of urgency.",
            },
            "informative": {
                "fr": "Informative and educational tone. Provide clear and factual details.",
                "en": "Informative and educational tone. Provide clear and factual details.",
                "ar": "Informative and educational tone. Provide clear and factual details.",
            }
        }
        
        default = "Professional and natural tone."
        return tone_map.get(tone.lower(), {}).get(lang, default)
    
    def rewrite_texts_batch(
        self,
        id_text_pairs: List[Tuple[int, str]],
        section_context: str = "",
        node_types: List[str] = None,
        language: str = "fr",
        tone: str = "professional",
        **kwargs
    ) -> List[Tuple[int, str]]:
        """Rewrite texts using OpenAI."""
        # Similar implementation to Gemini but using OpenAI API
        payload_with_length = []
        node_types = node_types or ['other'] * len(id_text_pairs)
        for idx, (i, t) in enumerate(id_text_pairs):
            word_count = len(t.split())
            char_count = len(t)
            node_type = node_types[idx] if idx < len(node_types) else 'other'
            payload_with_length.append({
                "id": i,
                "text": t,
                "original_words": word_count,
                "original_chars": char_count,
                "node_type": node_type
            })
        
        has_lorem = any("lorem" in t.lower() or "ipsum" in t.lower() for _, t in id_text_pairs)
        context_note = f"\nSection Context: {section_context}" if section_context else ""
        lang_name = self._get_language_name(language)
        tone_instructions = self._get_tone_instructions(tone, language)
        
        system_prompt = f"""You are a professional web copywriter specialized in real estate. Write in {lang_name} with a {tone_instructions} tone. Always return valid JSON only."""
        
        user_prompt = f"""
Rewrite the provided texts in {lang_name} with a {tone_instructions} tone.
{"⚠️ CRITICAL: Replace lorem ipsum/placeholder with real content, BUT keep EXACTLY the same length (±10% maximum)." if has_lorem else ""}

Brand Context:
- Brand: {self.brand}
- City: {self.city}
- Area: {self.area}
- Phone: {self.phone}
{context_note}

Keywords:
- Primary: {self.primary_kw}
- Secondary: {", ".join(self.secondary_kws)}

CRITICAL RULES:
1) Keep same language ({lang_name} only)
2) EXACT LENGTH: ±10% maximum word count difference
3) COMPLETE SENTENCES: All sentences must be complete
4) For headers: ensure complete phrases
5) Return ONLY valid JSON: [{{"id": 1, "text": "..."}}]

Input JSON:
{json.dumps(payload_with_length, ensure_ascii=False)}
"""
        
        result = self._call_openai(user_prompt, system_prompt)
        
        # Parse and process (similar to Gemini)
        try:
            # Clean response
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            parsed = json.loads(result)
            out = []
            for obj in parsed:
                new_text = str(obj["text"])
                obj_id = int(obj["id"])
                
                # Apply same length constraints as Gemini
                original_text = next((t for i, t in id_text_pairs if i == obj_id), "")
                node_type = 'other'
                if node_types:
                    for idx, (i, t) in enumerate(id_text_pairs):
                        if i == obj_id and idx < len(node_types):
                            node_type = node_types[idx]
                            break
                
                is_header = node_type in ['header', 'title']
                
                if original_text:
                    original_words = len(original_text.split())
                    new_words = len(new_text.split())
                    
                    if is_header:
                        if new_words > original_words * 1.5:
                            words = new_text.split()
                            target_words = int(original_words * 1.3)
                            if target_words < len(words):
                                new_text = " ".join(words[:target_words])
                    else:
                        if new_words > original_words * 1.2:
                            sentence_endings = ['.', '!', '?']
                            last_sentence_end = -1
                            for ending in sentence_endings:
                                pos = new_text.rfind(ending)
                                if pos > last_sentence_end:
                                    last_sentence_end = pos
                            
                            if last_sentence_end > len(new_text) * 0.6:
                                new_text = new_text[:last_sentence_end + 1]
                            else:
                                words = new_text.split()
                                target_words = int(original_words * 1.1)
                                trimmed_words = words[:target_words]
                                new_text = " ".join(trimmed_words)
                                if original_text.endswith(".") and not new_text.endswith("."):
                                    new_text += "."
                
                out.append((obj_id, new_text))
            return out
        except Exception as e:
            logging.error(f"Failed to parse OpenAI response: {e}")
            logging.debug(f"Response was: {result[:500]}")
            raise
    
    def optimize_image_alts_batch(
        self,
        images: List[dict],
        language: str = "fr",
        **kwargs
    ) -> List[dict]:
        """Optimize image alt texts using OpenAI."""
        lang_name = self._get_language_name(language)
        
        system_prompt = f"You are a real estate SEO expert. Return only valid JSON."
        
        user_prompt = f"""
Propose ALT attributes in {lang_name} for these images. For placeholder images, suggest search queries.

Return ONLY JSON: [{{"id": 1, "alt": "...", "image_query": "..." or null}}]

Images:
{json.dumps(images, ensure_ascii=False)}
"""
        
        result = self._call_openai(user_prompt, system_prompt)
        
        try:
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            parsed = json.loads(result)
            return parsed
        except Exception as e:
            logging.error(f"Failed to parse OpenAI alt response: {e}")
            raise


def create_ai_provider(
    provider: str,
    api_key: str,
    brand: str,
    city: str,
    area: str,
    phone: str,
    primary_kw: str,
    secondary_kws: List[str],
    **kwargs
) -> AIProvider:
    """Factory function to create AI provider."""
    provider = provider.lower()
    
    if provider == "gemini":
        return GeminiProvider(
            api_key=api_key,
            brand=brand,
            city=city,
            area=area,
            phone=phone,
            primary_kw=primary_kw,
            secondary_kws=secondary_kws,
            **kwargs
        )
    elif provider == "openai":
        return OpenAIProvider(
            api_key=api_key,
            brand=brand,
            city=city,
            area=area,
            phone=phone,
            primary_kw=primary_kw,
            secondary_kws=secondary_kws,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown AI provider: {provider}. Supported: 'gemini', 'openai'")
