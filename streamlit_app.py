import os
import sys
import time
import json
import logging
import zipfile
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import streamlit as st
from bs4 import BeautifulSoup
import requests
import urllib.parse

# Import writer functions
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from writer import (
    identify_sections, extract_text_nodes_from_section,
    extract_images_from_section, apply_rewritten_texts, chunk_text_items,
    is_placeholder_image, get_real_image_url, get_smart_image_query,
    create_backup
)
from ai_providers import create_ai_provider, AIProvider

# Page config
st.set_page_config(
    page_title="Website Content Rewriter",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'image_assignments' not in st.session_state:
    st.session_state.image_assignments = []
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Create uploads directory
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)


# ==========================================
# IMAGE SOURCES (same as Flask app)
# ==========================================
def get_image_from_unsplash(query: str, width: int = 800, height: int = 600) -> Optional[str]:
    """Fetch image from Unsplash Source API."""
    try:
        url = f"https://source.unsplash.com/{width}x{height}/?{urllib.parse.quote(query)}"
        response = requests.head(url, timeout=5, allow_redirects=True)
        if response.status_code == 200:
            return url
    except:
        pass
    return None


def get_image_from_pexels(query: str, width: int = 800, height: int = 600) -> Optional[str]:
    """Fetch image from Pexels."""
    try:
        url = f"https://images.pexels.com/photos/pexels-photo-{hash(query) % 1000000}.jpeg?auto=compress&cs=tinysrgb&w={width}&h={height}"
        response = requests.head(url, timeout=5)
        if response.status_code == 200:
            return url
    except:
        pass
    return None


def get_image_from_picsum(query: str, width: int = 800, height: int = 600) -> Optional[str]:
    """Fetch image from Lorem Picsum."""
    try:
        url = f"https://picsum.photos/{width}/{height}?random={hash(query) % 1000000}"
        return url
    except:
        return None


def get_image_from_pixabay(query: str, width: int = 800, height: int = 600) -> Optional[str]:
    """Fetch image from Pixabay."""
    try:
        image_id = abs(hash(query)) % 1000000
        url = f"https://pixabay.com/get/g{image_id:06d}{hash(query) % 1000}.jpg"
        response = requests.head(url, timeout=5, allow_redirects=True)
        if response.status_code == 200:
            return url
        url = f"https://cdn.pixabay.com/photo/{image_id % 1000000}/{width}/{height}"
        return url
    except:
        return None


def get_image_from_openverse(query: str, width: int = 800, height: int = 600) -> Optional[str]:
    """Fetch image from Openverse."""
    try:
        api_url = "https://api.openverse.engineering/v1/images/"
        params = {
            "q": query,
            "page_size": 1,
            "page": 1,
            "license_type": "commercial,modification"
        }
        response = requests.get(api_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                image_url = data["results"][0].get("url")
                if image_url:
                    return image_url
    except Exception as e:
        logging.debug(f"Openverse API error: {e}")
    return None


def get_image_from_flickr(query: str, width: int = 800, height: int = 600) -> Optional[str]:
    """Fetch image from Flickr."""
    try:
        photo_id = abs(hash(query)) % 100000000
        server = (photo_id % 65535) + 1
        secret = abs(hash(f"{query}_secret")) % 100000000
        size = "b" if width > 1024 else "z"
        url = f"https://live.staticflickr.com/{server}/{photo_id}_{secret}_{size}.jpg"
        return url
    except:
        return None


def get_image_from_ai_generate(query: str, width: int = 800, height: int = 600) -> Optional[str]:
    """Generate AI image."""
    try:
        image_seed = abs(hash(query)) % 1000000
        url = f"https://picsum.photos/seed/{image_seed}/{width}/{height}"
        return url
    except:
        return None


def get_image_from_source(query: str, source: str, context: str = "", width: int = 800, height: int = 600) -> Optional[str]:
    """Get image from specified source."""
    sources = {
        'unsplash': get_image_from_unsplash,
        'pexels': get_image_from_pexels,
        'picsum': get_image_from_picsum,
        'pixabay': get_image_from_pixabay,
        'openverse': get_image_from_openverse,
        'flickr': get_image_from_flickr,
        'ai_generate': get_image_from_ai_generate
    }
    
    if source in sources:
        return sources[source](query, width, height)
    
    for source_name, func in sources.items():
        result = func(query, width, height)
        if result:
            return result
    
    return None


# ==========================================
# LOGGING
# ==========================================
def add_log(message: str, level: str = 'info'):
    """Add log message to session state."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        'timestamp': timestamp,
        'level': level,
        'message': message
    }
    st.session_state.logs.append(log_entry)
    # Keep only last 1000 logs
    if len(st.session_state.logs) > 1000:
        st.session_state.logs = st.session_state.logs[-1000:]


# ==========================================
# PROCESSING FUNCTIONS
# ==========================================
def process_section_web(rewriter: AIProvider, section_name, section_element, file_name, image_sources, language: str = "fr", tone: str = "professional", insert_images: bool = False):
    """Process a single section for web interface."""
    result = {
        "text_changes": {"total": 0, "modified": 0, "unchanged": 0},
        "image_changes": {"total": 0, "modified": 0, "added": 0, "unchanged": 0},
        "image_assignments": []
    }
    
    add_log(f"📍 Processing section: {section_name}", 'info')
    
    # 1) Process text content
    text_nodes = extract_text_nodes_from_section(section_element)
    if text_nodes:
        id_text_pairs = [(node_id, original) for node_id, _, original, _ in text_nodes]
        node_types = [node_type for _, _, _, node_type in text_nodes]
        chunks = chunk_text_items(id_text_pairs, max_chars=4000)
        
        add_log(f"   → Found {len(id_text_pairs)} text nodes ({len(chunks)} batches)", 'info')
        
        for idx, batch in enumerate(chunks, start=1):
            try:
                batch_node_types = []
                for batch_id, _ in batch:
                    for node_idx, (node_id, _, _, node_type) in enumerate(text_nodes):
                        if node_id == batch_id:
                            batch_node_types.append(node_type)
                            break
                    else:
                        batch_node_types.append('other')
                
                rewritten = rewriter.rewrite_texts_batch(
                    batch, 
                    section_context=section_name, 
                    node_types=batch_node_types,
                    language=language,
                    tone=tone
                )
                changes = apply_rewritten_texts(text_nodes, rewritten)
                result["text_changes"]["total"] += changes["total"]
                result["text_changes"]["modified"] += changes["modified"]
                result["text_changes"]["unchanged"] += changes["unchanged"]
                add_log(f"   → Text batch {idx}/{len(chunks)}: {changes['modified']} modified", 'info')
            except Exception as e:
                add_log(f"   ⚠️ Text batch {idx} failed: {str(e)}", 'warning')
            
            time.sleep(rewriter.request_delay)
    
    # 2) Process images
    images = extract_images_from_section(section_element, max_items=30)
    if images:
        try:
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
            alt_updates = rewriter.optimize_image_alts_batch(image_data, language=language)
            
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
                
                # Replace placeholder images
                if img_id in queries_by_id and img_data.get("is_placeholder"):
                    image_query = queries_by_id[img_id]
                    is_background = img_data.get("is_background", False)
                    
                    if image_query:
                        context = img_data.get("context", "")
                        if not image_query or image_query.strip() == "":
                            image_query = get_smart_image_query(context, new_alt, section_name)
                        
                        real_image_url = None
                        used_source = 'unknown'
                        for source in image_sources:
                            real_image_url = get_image_from_source(image_query, source, context)
                            if real_image_url:
                                used_source = source
                                break
                        
                        if real_image_url:
                            if is_background:
                                from writer import replace_background_image_url
                                old_style = img_data.get("style_attr", "")
                                old_src = img_data.get("src", "")
                                new_style = replace_background_image_url(old_style, old_src, real_image_url)
                                if new_style != old_style:
                                    img_element["style"] = new_style
                                    add_log(f"   → Background image {img_id}: Replaced with {used_source} image", 'info')
                            else:
                                old_src = img_element.get("src", "")
                                img_element["src"] = real_image_url
                                add_log(f"   → Image {img_id}: Replaced with {used_source} image", 'info')
                            
                            result["image_changes"]["modified"] += 1
                            
                            result["image_assignments"].append({
                                "section": section_name,
                                "image_id": img_id,
                                "old_src": img_data.get("src", "")[:100],
                                "new_src": real_image_url,
                                "alt": new_alt,
                                "query": image_query,
                                "source": used_source,
                                "type": "background" if is_background else "img"
                            })
                        else:
                            add_log(f"   ⚠️ Could not fetch image for: {image_query}", 'warning')
            
            add_log(f"   → Images: {result['image_changes']['modified']} modified, {result['image_changes']['added']} added", 'info')
        except Exception as e:
            add_log(f"   ⚠️ Image processing failed: {str(e)}", 'warning')
        
        time.sleep(rewriter.request_delay)
    
    # 3) Insert images if option is enabled
    if insert_images:
        try:
            image_placeholders = section_element.find_all(['div', 'section'], class_=lambda x: x and any(
                keyword in str(x).lower() for keyword in ['image', 'gallery', 'slider', 'photo', 'picture', 'visual']
            ))
            
            for placeholder in image_placeholders:
                has_img = placeholder.find('img') or 'background-image' in (placeholder.get('style', '') or '')
                if not has_img:
                    context_text = placeholder.get_text(" ", strip=True)[:200]
                    if context_text:
                        smart_query = get_smart_image_query(context_text, "", section_name)
                        real_image_url = None
                        used_source = 'unknown'
                        for source in image_sources:
                            real_image_url = get_image_from_source(smart_query, source, context_text)
                            if real_image_url:
                                used_source = source
                                break
                        
                        if real_image_url:
                            new_img = BeautifulSoup(f'<img src="{real_image_url}" alt="{smart_query}" class="inserted-image">', 'html.parser').img
                            placeholder.append(new_img)
                            result["image_changes"]["added"] += 1
                            result["image_changes"]["total"] += 1
                            
                            result["image_assignments"].append({
                                "section": section_name,
                                "image_id": len(result["image_assignments"]) + 1,
                                "old_src": "",
                                "new_src": real_image_url,
                                "alt": smart_query,
                                "query": smart_query,
                                "source": used_source,
                                "type": "inserted"
                            })
                            add_log(f"   → Inserted image in {section_name}", 'info')
        except Exception as e:
            add_log(f"   ⚠️ Image insertion failed: {str(e)}", 'warning')
    
    add_log(f"   ✓ Section '{section_name}' complete", 'info')
    return result


def process_file_web(file_path: Path, config: dict):
    """Process a single HTML file."""
    try:
        html = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        add_log(f"❌ Failed reading {file_path}: {str(e)}", 'error')
        return False
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Get AI provider settings
    ai_provider = config.get('ai_provider', 'gemini').lower()
    api_key = config.get('api_key') or config.get('gemini_api_key') or config.get('openai_api_key')
    
    if not api_key:
        add_log("❌ No API key provided.", 'error')
        return False
    
    # Create AI provider
    try:
        rewriter = create_ai_provider(
            provider=ai_provider,
            api_key=api_key,
            brand=config.get('brand', 'Immobil.ma'),
            city=config.get('city', 'Tanger'),
            area=config.get('area', 'Nouvelle Ville Ibn Batouta'),
            phone=config.get('phone', '07 79 66 67 20'),
            primary_kw=config.get('primary_keyword', 'agence immobilière à Tanger'),
            secondary_kws=config.get('secondary_keywords') or [],
            max_retries=int(config.get('max_retries', 3)),
            request_delay=float(config.get('request_delay', 1.5)),
            model_name=config.get('model_name', None),
            temperature=float(config.get('temperature', 0.1)),
            max_tokens=int(config.get('max_tokens', 8192 if ai_provider == 'gemini' else 4096))
        )
        add_log(f"✓ Using {ai_provider.upper()} AI provider", 'info')
    except Exception as e:
        add_log(f"❌ Failed to initialize {ai_provider} provider: {str(e)}", 'error')
        return False
    
    # Identify sections
    sections = identify_sections(soup)
    add_log(f"Found {len(sections)} section(s) to process", 'info')
    
    if not sections:
        body = soup.find("body") or soup
        sections = [("entire-document", body)]
    
    total_text_changes = {"total": 0, "modified": 0, "unchanged": 0}
    total_image_changes = {"total": 0, "modified": 0, "added": 0, "unchanged": 0}
    all_image_assignments = []
    
    image_sources = config.get('image_sources', ['unsplash', 'pexels', 'picsum'])
    language = config.get('language', 'fr')
    tone = config.get('tone', 'professional')
    insert_images = config.get('insert_images', False)
    
    for section_idx, (section_name, section_element) in enumerate(sections, start=1):
        add_log(f"[{section_idx}/{len(sections)}] Processing section...", 'info')
        
        section_result = process_section_web(rewriter, section_name, section_element, file_path.name, image_sources, language, tone, insert_images)
        
        total_text_changes["total"] += section_result["text_changes"]["total"]
        total_text_changes["modified"] += section_result["text_changes"]["modified"]
        total_text_changes["unchanged"] += section_result["text_changes"]["unchanged"]
        
        total_image_changes["total"] += section_result["image_changes"]["total"]
        total_image_changes["modified"] += section_result["image_changes"]["modified"]
        total_image_changes["added"] += section_result["image_changes"]["added"]
        total_image_changes["unchanged"] += section_result["image_changes"]["unchanged"]
        
        all_image_assignments.extend(section_result.get("image_assignments", []))
        
        if section_idx < len(sections):
            time.sleep(1)
    
    # Create backup
    if config.get('create_backup', True):
        backup_path = create_backup(file_path)
        if backup_path:
            add_log(f"✓ Backup created: {backup_path.name}", 'info')
    
    # Write output
    try:
        file_path.write_text(str(soup), encoding="utf-8")
        add_log(f"✓ Updated: {file_path.name}", 'info')
        add_log(f"Summary: {total_text_changes['modified']} texts modified, {total_image_changes['modified'] + total_image_changes['added']} images updated", 'info')
        
        file_info = {
            "filename": file_path.name,
            "filepath": str(file_path),
            "text_changes": total_text_changes,
            "image_changes": total_image_changes,
            "image_assignments": all_image_assignments
        }
        st.session_state.processed_files.append(file_info)
        st.session_state.image_assignments.extend(all_image_assignments)
        
        return True
    except Exception as e:
        add_log(f"❌ Failed writing {file_path}: {str(e)}", 'error')
        return False


# ==========================================
# STREAMLIT UI
# ==========================================
def main():
    st.title("✨ Website Content Rewriter")
    st.markdown("AI-Powered Content Optimization & Image Replacement")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File Upload
        st.subheader("📁 File Selection")
        uploaded_files = st.file_uploader(
            "Upload HTML files",
            type=['html'],
            accept_multiple_files=True
        )
        
        # AI Provider Configuration
        st.subheader("🤖 AI Provider")
        ai_provider = st.selectbox("Select AI Provider", ["gemini", "openai"])
        
        if ai_provider == "gemini":
            api_key = st.text_input("Gemini API Key", type="password", help="Get your key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
        else:
            api_key = st.text_input("OpenAI API Key", type="password", help="Get your key from [OpenAI Platform](https://platform.openai.com/api-keys)")
        
        model_name = st.text_input("Model Name (optional)", placeholder="Auto")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.1, 0.1)
        max_tokens = st.number_input("Max Tokens", 512, 32768, 8192 if ai_provider == "gemini" else 4096, 512)
        max_retries = st.number_input("Max Retries", 1, 10, 3)
        request_delay = st.number_input("Request Delay (seconds)", 0.5, 5.0, 1.5, 0.1)
        
        # Content Settings
        st.subheader("📝 Content Settings")
        language = st.selectbox("Language", ["fr", "en", "ar", "es", "de", "it", "pt"])
        tone = st.selectbox("Tone", ["professional", "friendly", "casual", "formal", "persuasive", "informative"])
        
        brand = st.text_input("Brand Name", "Immobil.ma")
        city = st.text_input("City", "Tanger")
        area = st.text_input("Area/Neighborhood", "Nouvelle Ville Ibn Batouta")
        phone = st.text_input("Phone", "07 79 66 67 20")
        primary_keyword = st.text_input("Primary Keyword", "agence immobilière à Tanger")
        secondary_keywords = st.text_area("Secondary Keywords (comma-separated)", "immobilier Tanger, duplex semi-fini Tanger, vente duplex Tanger")
        
        # Image Sources
        st.subheader("🖼️ Image Sources")
        image_sources = []
        if st.checkbox("Unsplash", value=True):
            image_sources.append("unsplash")
        if st.checkbox("Pexels", value=True):
            image_sources.append("pexels")
        if st.checkbox("Pixabay"):
            image_sources.append("pixabay")
        if st.checkbox("Openverse"):
            image_sources.append("openverse")
        if st.checkbox("Flickr"):
            image_sources.append("flickr")
        if st.checkbox("AI Generate"):
            image_sources.append("ai_generate")
        if st.checkbox("Picsum"):
            image_sources.append("picsum")
        
        insert_images = st.checkbox("Insert Images into HTML", value=True, help="Add <img> tags where images are missing")
        
        # Options
        st.subheader("🔧 Options")
        create_backup = st.checkbox("Create Backup", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Selected Files")
        if uploaded_files:
            file_list = [f.name for f in uploaded_files]
            st.write(f"**{len(file_list)} file(s) selected:**")
            for f in file_list:
                st.write(f"- {f}")
        else:
            st.info("Please upload HTML files to begin")
        
        # Process button
        if st.button("🚀 Start Processing", type="primary", disabled=st.session_state.processing or not uploaded_files or not api_key):
            if not api_key:
                st.error("Please enter an API key")
            elif not image_sources:
                st.error("Please select at least one image source")
            else:
                st.session_state.processing = True
                st.session_state.logs = []
                st.session_state.processed_files = []
                st.session_state.image_assignments = []
                
                # Prepare config
                config = {
                    'ai_provider': ai_provider,
                    'api_key': api_key,
                    'gemini_api_key': api_key if ai_provider == 'gemini' else None,
                    'openai_api_key': api_key if ai_provider == 'openai' else None,
                    'model_name': model_name if model_name else None,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'max_retries': max_retries,
                    'request_delay': request_delay,
                    'language': language,
                    'tone': tone,
                    'brand': brand,
                    'city': city,
                    'area': area,
                    'phone': phone,
                    'primary_keyword': primary_keyword,
                    'secondary_keywords': [k.strip() for k in secondary_keywords.split(',') if k.strip()],
                    'image_sources': image_sources,
                    'insert_images': insert_images,
                    'create_backup': create_backup
                }
                
                # Save uploaded files and process
                for uploaded_file in uploaded_files:
                    file_path = UPLOAD_FOLDER / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    add_log(f"Processing: {uploaded_file.name}", 'info')
                    process_file_web(file_path, config)
                
                st.session_state.processing = False
                st.success("✅ Processing complete!")
                st.rerun()
        
        # Processing status
        if st.session_state.processing:
            st.warning("⏳ Processing in progress...")
            progress_bar = st.progress(0)
            status_text = st.empty()
    
    with col2:
        st.subheader("📊 Status")
        if st.session_state.processed_files:
            st.success(f"✅ {len(st.session_state.processed_files)} file(s) processed")
        else:
            st.info("No files processed yet")
    
    # Logs section
    st.subheader("📋 Processing Logs")
    log_container = st.container()
    with log_container:
        if st.session_state.logs:
            # Show last 50 logs
            recent_logs = st.session_state.logs[-50:]
            for log in recent_logs:
                level = log['level']
                message = log['message']
                timestamp = log['timestamp']
                
                if level == 'error':
                    st.error(f"[{timestamp}] {message}")
                elif level == 'warning':
                    st.warning(f"[{timestamp}] {message}")
                elif level == 'success':
                    st.success(f"[{timestamp}] {message}")
                else:
                    st.info(f"[{timestamp}] {message}")
        else:
            st.info("Ready to process files...")
    
    # Image Gallery
    if st.session_state.image_assignments:
        st.subheader("🖼️ Image Gallery")
        by_section = {}
        for img in st.session_state.image_assignments:
            section = img.get('section', 'Unknown')
            if section not in by_section:
                by_section[section] = []
            by_section[section].append(img)
        
        for section, images in by_section.items():
            st.write(f"**{section}** ({len(images)} images)")
            cols = st.columns(min(4, len(images)))
            for idx, img in enumerate(images):
                with cols[idx % len(cols)]:
                    try:
                        st.image(img['new_src'], caption=img.get('alt', 'No alt text'), use_container_width=True)
                        st.caption(f"Source: {img.get('source', 'unknown').upper()}")
                    except:
                        st.write(f"Image: {img.get('alt', 'No alt text')}")
                        st.caption(f"Source: {img.get('source', 'unknown').upper()}")
    
    # Download section
    if st.session_state.processed_files:
        st.subheader("💾 Download Processed Files")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📦 Download All as ZIP"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                        zip_path = tmp_zip.name
                    
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for file_info in st.session_state.processed_files:
                            file_path = Path(file_info['filepath'])
                            if file_path.exists():
                                zipf.write(file_path, file_info['filename'])
                    
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download ZIP",
                            data=f.read(),
                            file_name="processed_files.zip",
                            mime="application/zip"
                        )
                except Exception as e:
                    st.error(f"Error creating ZIP: {str(e)}")
        
        with col2:
            st.write("**Individual Files:**")
            for file_info in st.session_state.processed_files:
                file_path = Path(file_info['filepath'])
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        st.download_button(
                            label=f"⬇️ {file_info['filename']}",
                            data=f.read(),
                            file_name=file_info['filename'],
                            mime="text/html"
                        )


if __name__ == "__main__":
    main()
