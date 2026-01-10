import os
import sys
import time
import json
import logging
import argparse
import shutil
import requests
import urllib.parse
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

from bs4 import BeautifulSoup, NavigableString, Comment
import google.generativeai as genai


# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ==========================================
# DEFAULT SEO CONFIG (French-only real estate)
# ==========================================
DEFAULT_PRIMARY_KEYWORD = "agence immobilière à Tanger"
DEFAULT_SECONDARY_KEYWORDS = [
    "immobilier Tanger",
    "duplex semi-fini Tanger",
    "Nouvelle Ville Ibn Batouta",
    "vente duplex Tanger",
    "courtier immobilier Tanger",
    "accompagnement achat immobilier Tanger",
]

DEFAULT_CITY = "Tanger"
DEFAULT_AREA = "Nouvelle Ville Ibn Batouta"
DEFAULT_BRAND = "Immobil.ma"
DEFAULT_PHONE = "07 79 66 67 20"

MODEL_NAME = "gemini-2.0-flash"
GENERATION_CONFIG = {
    "temperature": 0.1,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}


# ==========================================
# UTILS
# ==========================================
def is_probably_visible_text(text: str) -> bool:
    """Filter out whitespace-only, tiny fragments, or junk."""
    if not text:
        return False
    t = text.strip()
    if len(t) < 2:
        return False
    # Avoid rewriting obvious code-ish fragments
    if "<" in t and ">" in t:
        return False
    return True


def should_skip_tag(tag_name: str) -> bool:
    return tag_name.lower() in {
        "script", "style", "noscript", "template", "svg", "math",
        "code", "pre"
    }


def chunk_text_items(items: List[Tuple[int, str]], max_chars: int = 4500) -> List[List[Tuple[int, str]]]:
    """
    Chunk list of (id, text) pairs by total char size, keeping each chunk under max_chars.
    """
    chunks = []
    current = []
    current_len = 0

    for item_id, txt in items:
        size = len(txt)
        if current and current_len + size > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append((item_id, txt))
        current_len += size

    if current:
        chunks.append(current)

    return chunks


# ==========================================
# GEMINI CLIENT
# ==========================================
class GeminiSEORewriter:
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
        request_delay: float = 1.5
    ):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(MODEL_NAME)
        self.brand = brand
        self.city = city
        self.area = area
        self.phone = phone
        self.primary_kw = primary_kw
        self.secondary_kws = secondary_kws
        self.max_retries = max_retries
        self.request_delay = request_delay

    def _call_gemini(self, prompt: str) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.model.generate_content(prompt, generation_config=GENERATION_CONFIG)
                text = (resp.text or "").strip()
                if text:
                    return text
            except Exception as e:
                logging.warning(f"Gemini error (attempt {attempt}/{self.max_retries}): {e}")
                time.sleep(2)
        raise RuntimeError("Gemini call failed after retries")

    def rewrite_texts_batch(self, id_text_pairs: List[Tuple[int, str]], section_context: str = "", node_types: List[str] = None) -> List[Tuple[int, str]]:
        """
        Sends a batch of texts to Gemini and returns rewritten (id, new_text).
        Output must remain French, professional, real estate oriented.
        Replaces lorem ipsum and placeholder text with real content.
        STRICTLY maintains original text length to preserve design.
        """
        # Calculate original lengths for each text
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
        
        # Check if there's lorem ipsum or placeholder text
        has_lorem = any("lorem" in t.lower() or "ipsum" in t.lower() for _, t in id_text_pairs)
        
        context_note = f"\nCONTEXTE SECTION: {section_context}" if section_context else ""
        
        prompt = f"""
Tu es un rédacteur web professionnel et expérimenté, spécialisé en immobilier au Maroc. Tu écris comme un humain expert, pas comme un robot.

OBJECTIF:
Réécris les textes fournis en FRANÇAIS avec un ton naturel, engageant et professionnel.
{"⚠️ CRITIQUE: Remplace le lorem ipsum/placeholder par du contenu réel et engageant, MAIS garde EXACTEMENT la même longueur (±10% maximum)." if has_lorem else ""}
Le contenu doit être naturel, fluide et convaincant - comme écrit par un expert immobilier qui connaît vraiment son métier.

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
- Écris comme un professionnel expérimenté, pas comme un robot
- Utilise un langage naturel et conversationnel mais professionnel
- Varie les phrases (courtes et longues) pour un rythme naturel
- Utilise des mots précis et engageants
- Évite les répétitions et les formules génériques
- Sois convaincant et rassurant, comme un vrai expert

RÈGLES STRICTES (CRITIQUES POUR PRÉSERVER LE DESIGN):
1) Garde la même langue (FR uniquement).
2) ⚠️ LONGUEUR EXACTE: Chaque texte de sortie doit avoir approximativement le MÊME nombre de mots que l'original (±10% maximum).
   - Si l'original fait 20 mots, le nouveau doit faire entre 18-22 mots.
   - Si l'original fait 2 lignes, le nouveau doit faire 2 lignes (pas 12!).
   - Ne PAS ajouter de paragraphes supplémentaires.
   - Ne PAS développer ou expliquer davantage.
3) {"Si c'est du lorem ipsum, crée du contenu réel, engageant et naturel sur l'immobilier à {self.city}, MAIS garde la même longueur que l'original. Écris comme un vrai professionnel qui connaît le marché." if has_lorem else "Conserve le sens et la longueur approximative (±10% maximum), mais rends le texte plus naturel et engageant."}
4) ⚠️ CRITIQUE - PHRASES COMPLÈTES: TOUTES les phrases doivent être COMPLÈTES et se terminer correctement.
   - Si c'est un titre/header (h1-h6), la phrase doit être complète et cohérente, même si cela dépasse légèrement la longueur cible.
   - Si c'est un paragraphe, toutes les phrases doivent se terminer par un point, point d'exclamation ou point d'interrogation.
   - Ne JAMAIS couper une phrase au milieu - si tu dois ajuster la longueur, termine la phrase complète même si c'est légèrement plus long.
   - Pour les headers/titres: assure-toi que c'est une phrase complète et cohérente, pas juste un fragment comme "Biens Immobiliers à" (complète: "Biens Immobiliers à Tanger" ou "Nos Biens Immobiliers").
5) Ne change pas les chiffres, adresses, noms de marque, numéros de téléphone.
6) N'invente pas d'informations factuelles (surface, prix, statut légal, etc.) si ce n'est pas dans le texte original.
7) Ne rajoute pas de titres si ce n'est pas un titre, ne transforme pas une liste en paragraphe.
8) Retourne UNIQUEMENT du JSON valide: une liste d'objets {{ "id": ..., "text": ... }}.
9) Ne retourne pas de markdown.
10) Le contenu doit être crédible, professionnel ET naturel - comme écrit par un humain expert.
11) ⚠️ IMPORTANT: Respecte la structure originale (nombre de phrases, paragraphes). Si l'original est court, reste court!
12) Évite les phrases robotiques comme "Nous sommes fiers de..." ou "Notre entreprise vous offre...". Sois plus direct et naturel.

ENTRÉE (JSON avec longueurs originales et types de nœuds):
{json.dumps(payload_with_length, ensure_ascii=False)}

IMPORTANT - TYPES DE NŒUDS:
- "header" ou "title": Ce sont des titres/headers (h1-h6). Ils DOIVENT être des phrases complètes et cohérentes, même si cela dépasse légèrement la longueur cible.
- "paragraph": Paragraphes normaux. Toutes les phrases doivent être complètes.
- "other": Autres éléments. Assure-toi que les phrases sont complètes.

⚠️ RAPPEL CRITIQUE: Pour les headers/titres, si tu vois "node_type": "header" ou "title", assure-toi que le texte est une phrase COMPLÈTE et cohérente. Ne laisse jamais un titre incomplet comme "Biens Immobiliers à" - complète-le en "Biens Immobiliers à Tanger" ou "Nos Biens Immobiliers" ou similaire.
"""
        result = self._call_gemini(prompt)

        # Expect JSON list of objects
        try:
            parsed = json.loads(result)
            out = []
            for obj in parsed:
                new_text = str(obj["text"])
                obj_id = int(obj["id"])
                
                # Find original text to compare length
                original_text = next((t for i, t in id_text_pairs if i == obj_id), "")
                # Find node type for this text
                node_type = 'other'
                if node_types:
                    for idx, (i, t) in enumerate(id_text_pairs):
                        if i == obj_id and idx < len(node_types):
                            node_type = node_types[idx]
                            break
                is_header = node_type in ['header', 'title']
                
                if original_text:
                    original_words = len(original_text.split())
                    original_chars = len(original_text)
                    new_words = len(new_text.split())
                    new_chars = len(new_text)
                    
                    # For headers/titles: ensure complete phrases, be more lenient with length
                    if is_header:
                        # Headers must be complete phrases - don't trim aggressively
                        if new_words > original_words * 1.5:  # More lenient for headers
                            # Find last complete word boundary
                            words = new_text.split()
                            target_words = int(original_words * 1.3)  # Allow 30% more for headers
                            if target_words < len(words):
                                trimmed_words = words[:target_words]
                                new_text = " ".join(trimmed_words)
                            # Ensure it doesn't end with incomplete words like "à" or "de"
                            if new_text and new_text.strip()[-1] in ['à', 'de', 'du', 'des', 'le', 'la', 'les', 'un', 'une']:
                                # Try to complete the phrase
                                if len(words) > target_words:
                                    new_text = " ".join(words[:target_words + 1])
                        # Ensure header is a complete phrase
                        new_text = new_text.strip()
                        if new_text and not new_text.endswith((".", "!", "?", ":", ";", ",")):
                            # If it ends with a preposition, try to complete it
                            if new_text.split()[-1].lower() in ['à', 'de', 'du', 'des', 'le', 'la', 'les', 'un', 'une', 'pour', 'avec', 'sans']:
                                # This is incomplete - try to get more words
                                words = new_text.split()
                                if len(words) < len(original_text.split()) * 1.2:
                                    # Try to complete from original context or add a simple completion
                                    if self.city and new_text.endswith(" à"):
                                        new_text = new_text + " " + self.city
                                    elif new_text.endswith(" de"):
                                        new_text = new_text + " " + self.area
                                    else:
                                        # Just ensure it's not cut off mid-word
                                        pass
                    else:
                        # For regular text: trim if too long, but always at sentence boundaries
                        if new_words > original_words * 1.2:
                            # Find last complete sentence
                            sentence_endings = ['.', '!', '?']
                            last_sentence_end = -1
                            for ending in sentence_endings:
                                pos = new_text.rfind(ending)
                                if pos > last_sentence_end:
                                    last_sentence_end = pos
                            
                            if last_sentence_end > len(new_text) * 0.6:  # If sentence end is in last 40%
                                # Trim at sentence boundary
                                new_text = new_text[:last_sentence_end + 1]
                            else:
                                # No sentence end found, trim by words but ensure complete words
                                words = new_text.split()
                                target_words = int(original_words * 1.1)
                                trimmed_words = words[:target_words]
                                new_text = " ".join(trimmed_words)
                                # Add period if original had one
                                if original_text.endswith(".") and not new_text.endswith("."):
                                    new_text += "."
                            logging.debug(f"Trimmed text {obj_id}: {new_words} → {len(new_text.split())} words (original: {original_words})")
                        
                        # Also check character length as secondary check
                        if new_chars > original_chars * 1.3:
                            # Trim by characters but at sentence boundary
                            sentence_endings = ['.', '!', '?']
                            last_sentence_end = -1
                            for ending in sentence_endings:
                                pos = new_text.rfind(ending)
                                if pos > last_sentence_end and pos > len(new_text) * 0.7:
                                    last_sentence_end = pos
                            
                            if last_sentence_end > 0:
                                new_text = new_text[:last_sentence_end + 1]
                            else:
                                # Fallback: trim but ensure complete words
                                trimmed = new_text[:int(original_chars * 1.2)]
                                last_space = trimmed.rfind(' ')
                                if last_space > len(trimmed) * 0.8:
                                    new_text = trimmed[:last_space]
                                else:
                                    new_text = trimmed
                
                out.append((obj_id, new_text))
            return out
        except Exception:
            # If model returns extra text, try to extract JSON
            start = result.find("[")
            end = result.rfind("]")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(result[start:end+1])
                out = []
                for o in parsed:
                    new_text = str(o["text"])
                    obj_id = int(o["id"])
                    
                    # Apply same length checking with sentence completion
                    original_text = next((t for i, t in id_text_pairs if i == obj_id), "")
                    # Find node type for this text
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
                            # Headers: ensure complete phrases
                            if new_words > original_words * 1.5:
                                words = new_text.split()
                                target_words = int(original_words * 1.3)
                                if target_words < len(words):
                                    new_text = " ".join(words[:target_words])
                        else:
                            # Regular text: trim at sentence boundaries
                            if new_words > original_words * 1.2:
                                # Find last sentence end
                                last_period = max(new_text.rfind('.'), new_text.rfind('!'), new_text.rfind('?'))
                                if last_period > len(new_text) * 0.6:
                                    new_text = new_text[:last_period + 1]
                                else:
                                    words = new_text.split()
                                    target_words = int(original_words * 1.1)
                                    trimmed_words = words[:target_words]
                                    new_text = " ".join(trimmed_words)
                                    if original_text.endswith(".") and not new_text.endswith("."):
                                        new_text += "."
                    
                    out.append((obj_id, new_text))
                return out
            raise

    def optimize_image_alts_batch(self, images: List[dict]) -> List[dict]:
        """
        images: list of dicts {id, src, existing_alt, context}
        returns: list of dicts {id, alt}
        """
        prompt = f"""
Tu es un expert SEO immobilier et web design.

OBJECTIF:
1) Proposer des attributs ALT FRANÇAIS, descriptifs et naturels (comme un humain les décrirait)
2) Identifier les images placeholder/dummy et suggérer des chemins d'images réelles appropriées

ATTRIBUTS ALT:
- Décris l'image de façon naturelle et précise (comme un humain le ferait)
- Reste concis (idéalement 6 à 15 mots)
- Inclus "{self.city}" ou "{self.area}" seulement si pertinent et naturel
- Évite les formules génériques comme "image de..." ou "photo de..."
- Sois spécifique: "Appartement moderne avec vue sur mer à {self.city}" plutôt que "Image d'appartement"
- Ne pas inventer de détails non visibles (pas de prix, surface, etc.)

IMAGES PLACEHOLDER:
- Si l'image est un placeholder, génère une requête de recherche appropriée pour trouver une vraie image
- La requête doit être en anglais et décrire le type d'image nécessaire
- Exemples de requêtes: "modern apartment interior", "real estate agent professional", "luxury property morocco", "apartment building facade"
- Sois spécifique: pour un appartement → "modern apartment living room", pour une équipe → "professional real estate team"

CONTEXTE:
- Marque: {self.brand}
- Ville: {self.city}
- Zone: {self.area}

RÈGLES:
1) Français uniquement pour les ALT.
2) ALT naturels et descriptifs, pas robotiques.
3) Si contexte insuffisant, ALT neutre mais descriptif: "Bien immobilier à {self.city}"
4) Pour les placeholders, suggère un chemin d'image logique et réaliste.

Retourne UNIQUEMENT du JSON valide: une liste d'objets {{ "id": ..., "alt": ..., "image_query": ... }}.
- "alt": attribut ALT en français
- "image_query": requête de recherche en anglais pour trouver une vraie image (ex: "modern apartment interior morocco")
  Si l'image n'est pas un placeholder, "image_query" peut être null.

DONNÉES (JSON):
{json.dumps(images, ensure_ascii=False)}
"""
        result = self._call_gemini(prompt)

        try:
            parsed = json.loads(result)
            return [
                {
                    "id": int(o["id"]), 
                    "alt": str(o.get("alt", "")),
                    "image_query": o.get("image_query") if o.get("image_query") else None
                } 
                for o in parsed
            ]
        except Exception:
            start = result.find("[")
            end = result.rfind("]")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(result[start:end+1])
                return [
                    {
                        "id": int(o["id"]), 
                        "alt": str(o.get("alt", "")),
                        "image_query": o.get("image_query") if o.get("image_query") else None
                    } 
                    for o in parsed
                ]
            raise


# ==========================================
# SECTION IDENTIFICATION
# ==========================================
def identify_sections(soup: BeautifulSoup) -> List[Tuple[str, any]]:
    """
    Identify major content sections in the HTML.
    Returns list of (section_name, section_element) tuples.
    Filters out very small sections (like breadcrumbs, headers, footers).
    """
    sections = []
    seen_elements = set()
    
    # Skip small sections (breadcrumbs, headers, footers, etc.)
    skip_patterns = ["breadcrumb", "header", "footer", "nav", "menu", "sidebar", "widget"]
    
    def should_skip_section(name: str) -> bool:
        """Check if section should be skipped."""
        name_lower = name.lower()
        return any(pattern in name_lower for pattern in skip_patterns)
    
    def has_substantial_content(element) -> bool:
        """Check if element has substantial text content (more than 50 chars)."""
        text = element.get_text(strip=True)
        return len(text) > 50
    
    # Look for section tags
    for section in soup.find_all("section"):
        section_id = section.get("id", "")
        section_class = " ".join(section.get("class", []))
        name = section_id or section_class or "section-unknown"
        if name and not should_skip_section(name) and section not in seen_elements:
            if has_substantial_content(section):
                sections.append((name, section))
                seen_elements.add(section)
    
    # Look for divs with "-area" classes (common pattern)
    for div in soup.find_all("div", class_=lambda x: x and any("-area" in str(c).lower() for c in (x if isinstance(x, list) else [x]))):
        classes = div.get("class", [])
        area_class = next((c for c in classes if "-area" in c.lower()), None)
        if area_class and not should_skip_section(area_class) and div not in seen_elements:
            if has_substantial_content(div):
                sections.append((area_class, div))
                seen_elements.add(div)
    
    # Look for major content divs (with common class patterns)
    major_patterns = ["about", "services", "features", "testimonial", "gallery", "contact", "apartments", "pricing", "team", "blog", "content", "main"]
    for pattern in major_patterns:
        for div in soup.find_all("div", class_=lambda x: x and any(pattern.lower() in str(c).lower() for c in (x if isinstance(x, list) else [x]))):
            classes = div.get("class", [])
            match_class = next((c for c in classes if pattern.lower() in c.lower()), None)
            if match_class and div not in seen_elements:
                if has_substantial_content(div):
                    sections.append((match_class, div))
                    seen_elements.add(div)
    
    # If no sections found, create one big section with main content
    if not sections:
        main = soup.find("main")
        if main:
            sections.append(("main-content", main))
        else:
            body = soup.find("body")
            if body:
                sections.append(("body-content", body))
    
    return sections


def extract_text_nodes_from_section(section_element) -> List[Tuple[int, NavigableString, str, str]]:
    """
    Extract visible text nodes from a specific section.
    Returns: (node_id, element, text, node_type)
    node_type: 'header' (h1-h6), 'title', 'paragraph', 'other'
    """
    nodes = []
    node_id = 0
    
    for element in section_element.descendants:
        if isinstance(element, Comment):
            continue
        if isinstance(element, NavigableString):
            parent = element.parent
            if not parent or not getattr(parent, "name", None):
                continue
            if should_skip_tag(parent.name):
                continue
            
            txt = str(element)
            if not is_probably_visible_text(txt):
                continue
            
            # Determine node type
            parent_tag = parent.name.lower()
            if parent_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                node_type = 'header'
            elif parent_tag in ['title']:
                node_type = 'title'
            elif parent_tag in ['p', 'div', 'span', 'li']:
                node_type = 'paragraph'
            else:
                node_type = 'other'
            
            node_id += 1
            nodes.append((node_id, element, txt, node_type))
    
    return nodes


def is_placeholder_image(src: str) -> bool:
    """Check if an image source is a placeholder/dummy image."""
    if not src:
        return True
    src_lower = src.lower()
    placeholder_indicators = [
        "placeholder", "place-holder", "dummy", "example", "sample", 
        "temp", "test", "demo", "default", "coming-soon", "no-image",
        "image-placeholder", "img-placeholder", "photo-placeholder"
    ]
    # Check for placeholder keywords
    if any(indicator in src_lower for indicator in placeholder_indicators):
        return True
    
    # Check if it's a local/demo image path (not a full URL)
    # Local paths like "img/slider/slider_img01.jpg" should be replaced
    if not src.startswith(("http://", "https://", "//")):
        # It's a local path - treat as placeholder if it looks like demo/test content
        if any(keyword in src_lower for keyword in ["slider_img", "test", "demo", "sample", "img/", "./img/", "../img/"]):
            return True
    
    return False


def extract_background_images_from_section(section_element, max_items: int = 30) -> List[dict]:
    """
    Extract CSS background-image URLs from style attributes in a section.
    Returns list of dicts with id, src, element, style_attr, context.
    """
    import re
    images = []
    img_id = 0
    
    # Find all elements with style attributes
    for element in section_element.find_all(attrs={"style": True}):
        style_attr = element.get("style", "")
        if not style_attr:
            continue
        
        # Match background-image:url(...) patterns
        # Handles: background-image:url(img.jpg), background-image: url('img.jpg'), etc.
        pattern = r'background-image\s*:\s*url\s*\(["\']?([^"\'()]+)["\']?\)'
        matches = re.findall(pattern, style_attr, re.IGNORECASE)
        
        for match in matches:
            src = match.strip()
            if not src:
                continue
            
            # Get context from element
            parent_text = element.get_text(" ", strip=True)[:200]
            context = parent_text
            
            # Check if placeholder (is_placeholder_image already checks for local paths)
            is_placeholder = is_placeholder_image(src)
            
            img_id += 1
            images.append({
                "id": img_id,
                "src": src[:300],
                "existing_alt": "",  # Background images don't have alt text
                "context": context,
                "is_placeholder": is_placeholder,
                "element": element,  # Store reference to element
                "style_attr": style_attr,  # Store original style
                "is_background": True  # Flag to identify background images
            })
            
            if len(images) >= max_items:
                break
        
        if len(images) >= max_items:
            break
    
    return images


def replace_background_image_url(style_attr: str, old_url: str, new_url: str) -> str:
    """
    Replace a background-image URL in a style attribute string.
    """
    import re
    # Escape special regex characters in old_url
    escaped_old = re.escape(old_url)
    # Match the pattern and replace
    pattern = rf'background-image\s*:\s*url\s*\(["\']?{escaped_old}["\']?\)'
    replacement = f"background-image:url('{new_url}')"
    new_style = re.sub(pattern, replacement, style_attr, flags=re.IGNORECASE)
    return new_style


def get_real_image_url(query: str, context: str = "", width: int = 800, height: int = 600) -> Optional[str]:
    """
    Fetch a real image URL from Unsplash API based on query.
    Falls back to Picsum if Unsplash fails.
    """
    # Clean and prepare search query
    search_query = query.lower().strip()
    if not search_query:
        # Generate query from context
        if "appartement" in context.lower() or "apartment" in context.lower():
            search_query = "modern apartment interior"
        elif "team" in context.lower() or "agent" in context.lower():
            search_query = "real estate agent professional"
        elif "gallery" in context.lower() or "property" in context.lower():
            search_query = "luxury real estate"
        else:
            search_query = "real estate property"
    
    # Try Unsplash Source API (free, no key needed)
    try:
        unsplash_url = f"https://source.unsplash.com/{width}x{height}/?{urllib.parse.quote(search_query)}"
        
        # Verify the URL is accessible
        response = requests.head(unsplash_url, timeout=5, allow_redirects=True)
        if response.status_code == 200:
            logging.debug(f"Found Unsplash image for query: {search_query}")
            return unsplash_url
    except Exception as e:
        logging.debug(f"Unsplash failed: {e}")
    
    # Fallback to Picsum (real photos, just random but reliable)
    try:
        fallback_url = f"https://picsum.photos/{width}/{height}?random={hash(search_query) % 1000000}"
        logging.debug(f"Using Picsum fallback for: {search_query}")
        return fallback_url
    except Exception as e:
        logging.warning(f"Image service error: {e}")
        return None


def get_smart_image_query(context: str, alt_text: str, section_name: str) -> str:
    """
    Generate a smart search query for finding relevant images.
    """
    # Combine context clues
    query_parts = []
    
    # Check section name
    if "apartment" in section_name.lower() or "appartement" in section_name.lower():
        query_parts.append("modern apartment")
    elif "team" in section_name.lower() or "about" in section_name.lower():
        query_parts.append("real estate professional")
    elif "gallery" in section_name.lower():
        query_parts.append("luxury property")
    elif "service" in section_name.lower():
        query_parts.append("real estate service")
    
    # Check alt text
    if alt_text:
        alt_lower = alt_text.lower()
        if "appartement" in alt_lower or "apartment" in alt_lower:
            query_parts.append("apartment interior")
        if "vue" in alt_lower or "view" in alt_lower:
            query_parts.append("property view")
        if "moderne" in alt_lower or "modern" in alt_lower:
            query_parts.append("modern")
    
    # Check context
    if context:
        ctx_lower = context.lower()
        if "tanger" in ctx_lower or "morocco" in ctx_lower or "maroc" in ctx_lower:
            query_parts.append("morocco")
    
    # Default query
    if not query_parts:
        query_parts.append("real estate property")
    
    return " ".join(query_parts[:3])  # Limit to 3 keywords


def extract_images_from_section(section_element, max_items: int = 30) -> List[dict]:
    """
    Extract images from a specific section (both <img> tags and CSS background-images).
    """
    images = []
    img_id = 0
    
    # 1) Extract <img> tags
    for img in section_element.find_all("img"):
        src = img.get("src", "") or ""
        existing_alt = img.get("alt", "") or ""
        
        # Nearby context: parent text (limited)
        parent_text = img.parent.get_text(" ", strip=True) if img.parent else ""
        context = parent_text[:200]
        
        # Check if placeholder
        is_placeholder = is_placeholder_image(src)
        
        img_id += 1
        images.append({
            "id": img_id,
            "src": src[:300],
            "existing_alt": existing_alt[:200],
            "context": context,
            "is_placeholder": is_placeholder,
            "element": img,  # Store reference to element
            "is_background": False
        })
        
        if len(images) >= max_items:
            break
    
    # 2) Extract CSS background-images (if we haven't reached max_items)
    if len(images) < max_items:
        bg_images = extract_background_images_from_section(section_element, max_items - len(images))
        # Adjust IDs to continue from where we left off
        for bg_img in bg_images:
            img_id += 1
            bg_img["id"] = img_id
            images.append(bg_img)
    
    return images


# ==========================================
# HTML PROCESSING (BeautifulSoup)
# ==========================================
def extract_visible_text_nodes(soup: BeautifulSoup) -> List[Tuple[int, NavigableString, str]]:
    """
    Returns list of (node_id, node, original_text) for visible text nodes.
    Skips scripts/styles/comments and common non-content tags.
    """
    nodes = []
    node_id = 0

    for element in soup.descendants:
        if isinstance(element, Comment):
            continue
        if isinstance(element, NavigableString):
            parent = element.parent
            if not parent or not getattr(parent, "name", None):
                continue
            if should_skip_tag(parent.name):
                continue

            txt = str(element)
            if not is_probably_visible_text(txt):
                continue

            node_id += 1
            nodes.append((node_id, element, txt))

    return nodes


def extract_images_for_alt(soup: BeautifulSoup, max_items: int = 50) -> List[dict]:
    """
    Collect <img> tags to optimize alt text.
    We create a small context based on nearby text.
    """
    images = []
    img_id = 0

    for img in soup.find_all("img"):
        src = img.get("src", "") or ""
        existing_alt = img.get("alt", "") or ""

        # Nearby context: parent text (limited)
        parent_text = img.parent.get_text(" ", strip=True) if img.parent else ""
        context = parent_text[:200]

        img_id += 1
        images.append({
            "id": img_id,
            "src": src[:300],
            "existing_alt": existing_alt[:200],
            "context": context
        })

        if len(images) >= max_items:
            break

    return images


def apply_rewritten_texts(text_nodes: List[Tuple[int, NavigableString, str, str]], rewritten: List[Tuple[int, str]]) -> dict:
    """
    Replaces the text in soup nodes by matching node_id.
    Returns statistics about changes made.
    text_nodes can be either (node_id, node, text) or (node_id, node, text, node_type)
    """
    mapping = {i: t for i, t in rewritten}
    changes = {"total": 0, "modified": 0, "unchanged": 0}
    
    for node_tuple in text_nodes:
        # Handle both old format (3 items) and new format (4 items)
        if len(node_tuple) == 4:
            node_id, node, original, _ = node_tuple
        else:
            node_id, node, original = node_tuple
        
        if node_id in mapping:
            new_text = mapping[node_id]
            changes["total"] += 1
            if new_text.strip() != original.strip():
                node.replace_with(new_text)
                changes["modified"] += 1
            else:
                changes["unchanged"] += 1
    
    return changes


def apply_optimized_alts(soup: BeautifulSoup, alt_updates: List[dict]) -> dict:
    """
    Applies alt text updates back onto the first N <img> tags in order of extraction.
    Returns statistics about changes made.
    """
    alts_by_id = {o["id"]: o["alt"] for o in alt_updates}
    img_id = 0
    changes = {"total": 0, "modified": 0, "added": 0, "unchanged": 0}
    
    for img in soup.find_all("img"):
        img_id += 1
        if img_id in alts_by_id:
            new_alt = alts_by_id[img_id]
            existing_alt = img.get("alt", "")
            changes["total"] += 1
            
            if not existing_alt:
                img["alt"] = new_alt
                changes["added"] += 1
            elif existing_alt.strip() != new_alt.strip():
                img["alt"] = new_alt
                changes["modified"] += 1
            else:
                changes["unchanged"] += 1
    
    return changes


def create_backup(file_path: Path) -> Optional[Path]:
    """Create a backup of the file with timestamp."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f".backup_{timestamp}{file_path.suffix}")
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        logging.warning(f"Failed to create backup: {e}")
        return None


def process_section(
    rewriter: GeminiSEORewriter,
    section_name: str,
    section_element,
    file_name: str
) -> dict:
    """
    Process a single section: update text content and images.
    Returns statistics.
    """
    result = {
        "text_changes": {"total": 0, "modified": 0, "unchanged": 0},
        "image_changes": {"total": 0, "modified": 0, "added": 0, "unchanged": 0}
    }
    
    logging.info(f"  📍 Section: {section_name}")
    
    # 1) Process text content in this section
    text_nodes = extract_text_nodes_from_section(section_element)
    if text_nodes:
        id_text_pairs = [(node_id, original) for node_id, _, original, _ in text_nodes]
        node_types = [node_type for _, _, _, node_type in text_nodes]
        chunks = chunk_text_items(id_text_pairs, max_chars=4000)
        
        logging.info(f"    → Found {len(id_text_pairs)} text nodes ({len(chunks)} batches)")
        
        for idx, batch in enumerate(chunks, start=1):
            try:
                # Get node types for this batch
                batch_node_types = []
                for batch_id, _ in batch:
                    # Find corresponding node type
                    for node_idx, (node_id, _, _, node_type) in enumerate(text_nodes):
                        if node_id == batch_id:
                            batch_node_types.append(node_type)
                            break
                    else:
                        batch_node_types.append('other')
                
                rewritten = rewriter.rewrite_texts_batch(batch, section_context=section_name, node_types=batch_node_types)
                changes = apply_rewritten_texts(text_nodes, rewritten)
                result["text_changes"]["total"] += changes["total"]
                result["text_changes"]["modified"] += changes["modified"]
                result["text_changes"]["unchanged"] += changes["unchanged"]
                logging.info(f"    → Text batch {idx}/{len(chunks)}: {changes['modified']} modified")
            except Exception as e:
                logging.warning(f"    ⚠️ Text batch {idx} failed: {e}")
            
            time.sleep(rewriter.request_delay)
    
    # 2) Process images in this section
    images = extract_images_from_section(section_element, max_items=30)
    if images:
        try:
            # Prepare image data (without element reference for JSON)
            image_data = [
                {
                    "id": img["id"], 
                    "src": img["src"], 
                    "existing_alt": img["existing_alt"], 
                    "context": img["context"],
                    "is_placeholder": img.get("is_placeholder", False)
                } 
                for img in images
            ]
            alt_updates = rewriter.optimize_image_alts_batch(image_data)
            
            # Apply alt updates and fetch real images for placeholders
            alts_by_id = {o["id"]: o.get("alt", "") for o in alt_updates}
            queries_by_id = {o["id"]: o.get("image_query") for o in alt_updates if o.get("image_query")}
            
            for img_data in images:
                img_id = img_data["id"]
                img_element = img_data["element"]
                
                # Update ALT text
                if img_id in alts_by_id:
                    new_alt = alts_by_id[img_id]
                    existing_alt = img_element.get("alt", "")
                    
                    result["image_changes"]["total"] += 1
                    if not existing_alt:
                        img_element["alt"] = new_alt
                        result["image_changes"]["added"] += 1
                    elif existing_alt.strip() != new_alt.strip():
                        img_element["alt"] = new_alt
                        result["image_changes"]["modified"] += 1
                    else:
                        result["image_changes"]["unchanged"] += 1
                
                # Replace placeholder image source with real image URL
                if img_id in queries_by_id and img_data.get("is_placeholder"):
                    image_query = queries_by_id[img_id]
                    old_src = img_element.get("src", "")
                    
                    if image_query:
                        # Get context for better image selection
                        context = img_data.get("context", "")
                        
                        # Generate smart query if needed
                        if not image_query or image_query.strip() == "":
                            image_query = get_smart_image_query(context, new_alt, section_name)
                        
                        # Fetch real image URL
                        real_image_url = get_real_image_url(image_query, context)
                        
                        if real_image_url:
                            img_element["src"] = real_image_url
                            logging.info(f"    → Image {img_id}: Replaced placeholder with real image")
                            logging.info(f"       Query: '{image_query}' → URL: {real_image_url[:80]}...")
                            result["image_changes"]["modified"] += 1
                        else:
                            logging.warning(f"    ⚠️ Could not fetch real image for query: {image_query}")
                    else:
                        # Fallback: use smart query generation
                        context = img_data.get("context", "")
                        smart_query = get_smart_image_query(context, new_alt, section_name)
                        real_image_url = get_real_image_url(smart_query, context)
                        if real_image_url:
                            img_element["src"] = real_image_url
                            logging.info(f"    → Image {img_id}: Replaced with smart query '{smart_query}'")
                            result["image_changes"]["modified"] += 1
            
            logging.info(f"    → Images: {result['image_changes']['modified']} modified, {result['image_changes']['added']} added")
        except Exception as e:
            logging.warning(f"    ⚠️ Image processing failed: {e}")
        
        time.sleep(rewriter.request_delay)
    
    logging.info(f"    ✓ Section '{section_name}' complete")
    return result


def process_html_file(
    rewriter: GeminiSEORewriter,
    file_path: Path,
    in_place: bool = True,
    backup: bool = True,
    dry_run: bool = False
) -> dict:
    """
    Process a single HTML file section by section:
    - Identify major sections
    - Process each section: rewrite text content and update images
    - Replace lorem ipsum and placeholder content
    
    Returns dict with processing results and statistics.
    """
    result = {
        "success": False,
        "text_changes": {},
        "image_changes": {},
        "backup_path": None,
        "error": None,
        "sections_processed": 0
    }
    
    try:
        html = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logging.error(f"Failed reading {file_path}: {e}")
        result["error"] = str(e)
        return result

    soup = BeautifulSoup(html, "html.parser")

    # Identify sections
    sections = identify_sections(soup)
    logging.info(f"{file_path.name}: Found {len(sections)} section(s) to process")
    
    if not sections:
        logging.warning(f"{file_path.name}: No sections identified, processing entire document")
        # Fallback: process entire document as one section
        body = soup.find("body") or soup
        sections = [("entire-document", body)]

    # Process each section sequentially
    total_text_changes = {"total": 0, "modified": 0, "unchanged": 0}
    total_image_changes = {"total": 0, "modified": 0, "added": 0, "unchanged": 0}
    
    for section_idx, (section_name, section_element) in enumerate(sections, start=1):
        logging.info("")
        logging.info(f"  [{section_idx}/{len(sections)}] Processing section...")
        
        section_result = process_section(rewriter, section_name, section_element, file_path.name)
        
        # Accumulate statistics
        total_text_changes["total"] += section_result["text_changes"]["total"]
        total_text_changes["modified"] += section_result["text_changes"]["modified"]
        total_text_changes["unchanged"] += section_result["text_changes"]["unchanged"]
        
        total_image_changes["total"] += section_result["image_changes"]["total"]
        total_image_changes["modified"] += section_result["image_changes"]["modified"]
        total_image_changes["added"] += section_result["image_changes"]["added"]
        total_image_changes["unchanged"] += section_result["image_changes"]["unchanged"]
        
        result["sections_processed"] += 1
        
        # Small delay between sections
        if section_idx < len(sections):
            time.sleep(1)

    result["text_changes"] = total_text_changes
    result["image_changes"] = total_image_changes

    # Write output
    if dry_run:
        logging.info(f"[DRY RUN] Would update: {file_path}")
        logging.info(f"  Text changes: {total_text_changes['modified']} modified, {total_text_changes['unchanged']} unchanged")
        logging.info(f"  Image changes: {total_image_changes['modified']} modified, {total_image_changes['added']} added")
        result["success"] = True
        return result

    # Create backup if requested
    if backup and in_place:
        backup_path = create_backup(file_path)
        if backup_path:
            result["backup_path"] = str(backup_path)
            logging.info(f"Backup created: {backup_path.name}")

    # Write to file (always in-place for this version)
    try:
        file_path.write_text(str(soup), encoding="utf-8")
        logging.info(f"✓ Updated: {file_path.name}")
        logging.info(f"  Summary: {total_text_changes['modified']} texts modified, {total_image_changes['modified'] + total_image_changes['added']} images updated")
        logging.info(f"  Sections processed: {result['sections_processed']}/{len(sections)}")
        result["success"] = True
    except Exception as e:
        logging.error(f"Failed writing {file_path}: {e}")
        result["error"] = str(e)

    return result


# ==========================================
# BATCH PROCESSING
# ==========================================
def iter_html_files(input_dir: Path, recursive: bool, target_files: Optional[List[str]] = None) -> List[Path]:
    """
    Get list of HTML files to process.
    If target_files is provided, only return files matching those names.
    When target_files is specified, always search recursively.
    """
    if target_files:
        # Target specific files - always search recursively when looking for specific files
        files = []
        target_set = {f.lower() for f in target_files}
        pattern = "**/*.html"  # Always recursive when searching for specific files
        
        for file_path in input_dir.glob(pattern):
            if file_path.name.lower() in target_set:
                files.append(file_path)
        
        # If not found, try checking if file exists as direct path
        if not files:
            for target_file in target_files:
                # Try as direct path
                direct_path = input_dir / target_file
                if direct_path.exists() and direct_path.suffix.lower() == ".html":
                    files.append(direct_path)
                # Try in html/ subdirectory (common location)
                html_subdir = input_dir / "html" / target_file
                if html_subdir.exists() and html_subdir.suffix.lower() == ".html":
                    files.append(html_subdir)
        
        return sorted(files)
    else:
        # All HTML files
        pattern = "**/*.html" if recursive else "*.html"
        return sorted(input_dir.glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description="SEO Content Rewriter - Optimizes text content and image alt attributes in HTML files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update index.html in current directory (section by section)
  python writer.py --file index.html

  # Update index.html in a specific directory
  python writer.py --input ./html --file index.html

  # Dry run to see what would change
  python writer.py --file index.html --dry-run

  # Update multiple files (one at a time, section by section)
  python writer.py --file index.html about.html
        """
    )
    
    parser.add_argument("--input", default=".", help="Input folder containing .html files (default: current directory)")
    parser.add_argument("--file", nargs="+", required=True, help="Target specific file(s) by name (REQUIRED: e.g., --file index.html)")
    parser.add_argument("--recursive", action="store_true", help="Search for HTML files recursively in subfolders")
    parser.add_argument("--no-in-place", action="store_true", help="Don't modify files in-place (creates output files instead)")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup files before modifying")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying files")

    parser.add_argument("--brand", default=DEFAULT_BRAND, help="Brand name for SEO context")
    parser.add_argument("--city", default=DEFAULT_CITY, help="City name for SEO context")
    parser.add_argument("--area", default=DEFAULT_AREA, help="Area/neighborhood for SEO context")
    parser.add_argument("--phone", default=DEFAULT_PHONE, help="Phone number for SEO context")

    parser.add_argument("--primary", default=DEFAULT_PRIMARY_KEYWORD, help="Primary SEO keyword")
    parser.add_argument("--secondary", default=",".join(DEFAULT_SECONDARY_KEYWORDS), help="Secondary keywords (comma-separated)")
    parser.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY environment variable)")

    args = parser.parse_args()

    # Try to get API key from: 1) command line, 2) environment variable, 3) .env file
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    
    # Try loading from .env file if not found
    if not api_key:
        # Look for .env file in project root (parent of html/ directory)
        env_file = Path(__file__).parent.parent / ".env"
        
        # Also check current working directory as fallback
        if not env_file.exists():
            env_file = Path.cwd() / ".env"
        
        if env_file.exists():
            try:
                logging.info(f"Loading API key from .env file: {env_file}")
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        # Skip comments and empty lines
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        
                        # Parse KEY=VALUE format
                        if "=" in line and line.startswith("GEMINI_API_KEY"):
                            # Split on first = only
                            key_part, value_part = line.split("=", 1)
                            # Remove quotes if present
                            api_key = value_part.strip().strip('"').strip("'").strip()
                            if api_key:
                                logging.info("✓ API key loaded from .env file")
                                break
            except Exception as e:
                logging.warning(f"Could not read .env file: {e}")
        else:
            logging.debug(f".env file not found at: {env_file}")
    if not api_key:
        logging.error("Missing GEMINI_API_KEY. Set it via:")
        logging.error("  1. Command line: --api-key YOUR_KEY")
        logging.error("  2. Environment variable: export GEMINI_API_KEY=YOUR_KEY (Linux/Mac) or set GEMINI_API_KEY=YOUR_KEY (Windows)")
        logging.error("  3. .env file: Create a .env file in the project root with: GEMINI_API_KEY=YOUR_KEY")
        sys.exit(1)

    secondary_kws = [k.strip() for k in args.secondary.split(",") if k.strip()]

    rewriter = GeminiSEORewriter(
        api_key=api_key,
        brand=args.brand,
        city=args.city,
        area=args.area,
        phone=args.phone,
        primary_kw=args.primary,
        secondary_kws=secondary_kws,
        max_retries=3,
        request_delay=1.5
    )

    input_dir = Path(args.input).resolve()
    
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Get files to process
    files = iter_html_files(input_dir, args.recursive, args.file)
    
    if not files:
        if args.file:
            logging.error(f"❌ Target file(s) not found: {', '.join(args.file)}")
            logging.error(f"   Searched in: {input_dir}")
            logging.error(f"   Also checked: {input_dir / 'html'}")
            logging.error("")
            logging.error("💡 Try one of these:")
            logging.error(f"   1. Specify full path: --input ./html --file index.html")
            logging.error(f"   2. Use relative path from script location")
            logging.error(f"   3. Check if file exists in a subdirectory")
        else:
            logging.warning(f"No .html files found in {input_dir}")
        sys.exit(1)

    # Determine if in-place editing
    in_place = not args.no_in_place
    
    # Log processing mode
    mode = "DRY RUN" if args.dry_run else ("IN-PLACE" if in_place else "OUTPUT")
    backup_mode = "WITH BACKUP" if (not args.no_backup and in_place and not args.dry_run) else "NO BACKUP"
    
    logging.info("=" * 60)
    logging.info(f"SEO Content Rewriter - Processing Mode: {mode} ({backup_mode})")
    logging.info(f"Input directory: {input_dir}")
    if args.file:
        logging.info(f"Target files: {', '.join(args.file)}")
    logging.info(f"Found {len(files)} HTML file(s) to process")
    logging.info("=" * 60)

    success = 0
    total_text_changes = {"modified": 0, "unchanged": 0}
    total_image_changes = {"modified": 0, "added": 0, "unchanged": 0}

    for fpath in files:
        logging.info("")
        logging.info(f"Processing: {fpath.name}")
        logging.info("-" * 60)
        
        result = process_html_file(
            rewriter=rewriter,
            file_path=fpath,
            in_place=in_place,
            backup=not args.no_backup,
            dry_run=args.dry_run
        )
        
        if result["success"]:
            success += 1
            total_text_changes["modified"] += result["text_changes"].get("modified", 0)
            total_text_changes["unchanged"] += result["text_changes"].get("unchanged", 0)
            total_image_changes["modified"] += result["image_changes"].get("modified", 0)
            total_image_changes["added"] += result["image_changes"].get("added", 0)
            total_image_changes["unchanged"] += result["image_changes"].get("unchanged", 0)
            
            if result.get("backup_path"):
                logging.info(f"  Backup: {Path(result['backup_path']).name}")

    # Final summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("PROCESSING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Files processed: {success}/{len(files)}")
    logging.info(f"Text changes: {total_text_changes['modified']} modified, {total_text_changes['unchanged']} unchanged")
    logging.info(f"Image changes: {total_image_changes['modified']} modified, {total_image_changes['added']} added, {total_image_changes['unchanged']} unchanged")
    
    # Show sections processed per file
    for fpath in files:
        logging.info(f"  {fpath.name}: Processed section by section")
    
    if args.dry_run:
        logging.info("")
        logging.info("This was a DRY RUN. No files were modified.")
        logging.info("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
