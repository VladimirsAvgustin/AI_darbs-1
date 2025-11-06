import os
import sys
from dotenv import load_dotenv
from typing import List

try:
    from huggingface_hub import InferenceClient
except Exception as e:
    print("Missing huggingface_hub. Install: pip install huggingface_hub python-dotenv")
    raise e

# Model configuration - you can replace these with any instruction-tuned models available on HF
SUMMARIZE_MODEL = "facebook/bart-large-cnn"      # Model for text summarization
TEXT2TEXT_MODEL = "google/flan-t5-large"         # Instruction-tuned model (if available)
FALLBACK_GEN_MODEL = "gpt2"                      # Fallback model for text generation (non-instruction)

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY", "").strip()
OFFLINE_FLAG = "--offline" in sys.argv or not HF_TOKEN


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {path}: {e}")

def init_hf() -> InferenceClient | None:
    if OFFLINE_FLAG:
        return None
    try:
        client = InferenceClient(token=HF_TOKEN)
        return client
    except Exception as e:
        print("Warning: HuggingFace client init failed:", e)
        return None

# ===== Summarization =====
def summarize_with_hf(client: InferenceClient | None, text: str) -> str:
    """Summarize text using HuggingFace model or offline fallback."""
    if not text:
        return ""
    if client is None:
        # Offline fallback: extract first meaningful paragraph
        # Clean and normalize text first
        text = ' '.join(text.strip().split())  # Remove extra whitespace
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return text[:100] + "..." if len(text) > 100 else text
            
        # Find the most meaningful paragraph (longest with good sentence structure)
        best_para = None
        best_score = 0
        
        for para in paragraphs:
            # Score based on length and sentence structure
            sentences = [s.strip() for s in para.split('.') if len(s.strip()) > 20]
            score = len(sentences) * sum(len(s) for s in sentences)
            if score > best_score:
                best_score = score
                best_para = para
        
        # Return the best paragraph, or fallback to text excerpt
        if best_para:
            return best_para
        return text[:100] + "..." if len(text) > 100 else text
    
    try:
        # Use facebook/bart-large-cnn with optimized parameters
        resp = client.summarization(
            text=text,
            model=SUMMARIZE_MODEL,
            parameters={
                "max_length": 80,      # Keep summary concise
                "min_length": 30,      # But not too short
                "length_penalty": 0.8,  # Prefer shorter summaries
                "num_beams": 4,        # More careful word choice
                "do_sample": False     # Deterministic output
            }
        )
        
        # Clean up the response
        if isinstance(resp, str):
            summary = resp.strip()
        elif isinstance(resp, (list, tuple)) and resp:
            first = resp[0]
            if isinstance(first, dict):
                summary = first.get("summary_text", "") or first.get("generated_text", "") or str(first)
            else:
                summary = str(first)
        elif hasattr(resp, "generated_text"):
            summary = getattr(resp, "generated_text").strip()
        else:
            summary = str(resp).strip()
        
        # Remove any raw output formatting
        if summary.startswith("SummarizationOutput"):
            summary = summary.replace("SummarizationOutput(summary_text=", "").rstrip(")")
            summary = summary.strip("'\"")
        
        return summary.strip()
    except Exception as e:
        # Try without parameters if they're not supported
        try:
            resp = client.summarization(text, model=SUMMARIZE_MODEL)
            summary = str(resp).strip()
            if summary.startswith("SummarizationOutput"):
                summary = summary.replace("SummarizationOutput(summary_text=", "").rstrip(")")
                summary = summary.strip("'\"")
            return summary
        except Exception as e2:
            return f"[summarization error] {e2}"

# ===== Keywords (HF) =====
def generate_keywords_with_hf(client: InferenceClient | None, text: str, count: int) -> List[str]:
    if not text:
        return []
    if client is None:
        import re
        # Clean and normalize text
        text = ' '.join(text.strip().split())
        
        # Define words we want to exclude (common words, etc)
        common_words = {
            'about', 'after', 'again', 'also', 'another', 'before', 'being', 'between',
            'both', 'could', 'doing', 'during', 'each', 'either', 'every', 'from',
            'having', 'here', 'might', 'more', 'most', 'much', 'must', 'only',
            'other', 'same', 'some', 'such', 'than', 'that', 'their', 'them',
            'then', 'there', 'these', 'they', 'this', 'through', 'very', 'what',
            'when', 'where', 'which', 'while', 'with', 'would', 'your'
        }
        
        # Extract words with better filtering
        words = []
        for word in text.split():
            # Clean the word
            word = re.sub(r'[^\w\s]', '', word)
            if word and len(word) > 4:  # Only meaningful words
                word_lower = word.lower()
                if (word_lower not in common_words and  # Not a common word
                    word.isalpha() and                  # Only letters
                    not word.isupper()):               # Not an acronym
                    words.append(word)
        
        # Get unique words preserving case
        seen = set()
        filtered = []
        for word in words:
            word_lower = word.lower()
            if word_lower not in seen:
                seen.add(word_lower)
                # Prefer capitalized versions of words
                filtered.append(word)
        
        # Sort by relevance (length and position in text)
        filtered.sort(key=lambda w: (-len(w), text.lower().find(w.lower())))
        
        return filtered[:count]
    try:
        prompt = f"Extract {count} descriptive keywords from the following text. Return a comma-separated list:\n\n{text}\n\nKeywords:"
        # Try instruction-tuned text2text_generation, if available
        try:
            resp = client.text2text_generation(prompt, model=TEXT2TEXT_MODEL, max_new_tokens=128)
        except Exception:
            resp = client.text_generation(prompt, model=FALLBACK_GEN_MODEL, max_new_tokens=128)
        # Normalize response
        out_text = ""
        if isinstance(resp, str):
            out_text = resp
        elif isinstance(resp, (list, tuple)) and resp:
            first = resp[0]
            if isinstance(first, dict):
                out_text = first.get("generated_text") or first.get("output_text") or str(first)
            else:
                out_text = str(first)
        elif hasattr(resp, "generated_text"):
            out_text = getattr(resp, "generated_text")
        else:
            out_text = str(resp)
        items = [it.strip() for it in out_text.replace("\n", ",").split(",") if it.strip()]
        return items[:count]
    except Exception as e:
        
        # Fallback to offline extraction
        return generate_keywords_with_hf(None, text, count)

# ===== Quiz (HF) =====
def generate_quiz_with_hf(client: InferenceClient | None, text: str, num_questions: int = 3) -> str:
    """Generate a multiple-choice quiz using HF models or offline fallback."""
    if not text:
        return ""
    
    if client is None:
        # Extract meaningful words (nouns, meaningful terms) from text
        import re
        import random  # For shuffling and selection
        text = text.lower()  # Normalize text
        # Split into words and clean them
        words = []
        for word in text.replace('\n', ' ').split():
            # Remove punctuation and clean
            word = re.sub(r'[.,!?()[\]{}":;]', '', word)
            # Keep only meaningful words (longer than 4 chars, no numbers)
            if word and len(word) > 4 and word.isalpha():
                # Skip common words and verbs that aren't meaningful for questions
                if word not in {
                    'about', 'these', 'their', 'there', 'which', 'what', 'when', 'where', 'would', 'could', 'should',
                    'being', 'doing', 'going', 'making', 'taking', 'using', 'working', 'becoming', 'getting', 'having',
                    'looking', 'seeing', 'using', 'going', 'make', 'take', 'work', 'find', 'give', 'know', 'think',
                    'come', 'look', 'want', 'been', 'were', 'have', 'into', 'some', 'than', 'then', 'from', 'that',
                    'this', 'will', 'with', 'they', 'your'
                }:
                    words.append(word)
        if len(words) < 8:  # Need enough words for varied questions
            return "Text too short to generate meaningful quiz."
        
        # Define varied question templates
        templates = [
            "Which term best represents a key concept in the text?",
            "What important topic or technology is discussed in detail?",
            "Which of these terms is most significant to the main subject?",
            "What concept plays a central role in the text?",
            "Which term represents a major focus of the discussion?",
            "What important element is described in the text?",
            "Which concept is fundamental to the text's topic?",
            "What key term is essential to understanding the content?"
        ]
        
        out = []
        used = set()
        
        # Find word clusters based on proximity in text
        word_clusters = {}
        window_size = 100  # Characters to look before/after each word
        
        for word in words:
            word_pos = text.find(word)
            if word_pos == -1:  # Skip if word not found (shouldn't happen)
                continue
                
            # Get nearby text window
            start = max(0, word_pos - window_size)
            end = min(len(text), word_pos + window_size)
            window = text[start:end].lower()
            
            # Group words that appear in similar context
            related_words = []
            for other in words:
                if other != word and other in window:
                    related_words.append(other)
            
            word_clusters[word] = related_words
        
        for i in range(min(num_questions, len(templates))):
            # Find words with enough related terms for good options
            available_words = [w for w in word_clusters.keys() 
                             if w not in used and len(set(word_clusters[w]) - used) >= 3]
            
            if not available_words:  # Fallback if we can't find related words
                available = [w for w in words if w not in used]
                if len(available) < 4:
                    break
                # Use random selection as fallback
                selected = random.sample(available, 4)
                opts = selected
                used.update(selected)
            else:
                # Pick a word and its related terms for more meaningful options
                target_word = random.choice(available_words)
                related = list(set(word_clusters[target_word]) - used)
                
                # Select 3 related words for options
                if len(related) >= 3:
                    wrong_options = random.sample(related, 3)
                else:
                    # Fill with other unused words if not enough related ones
                    wrong_options = related
                    other_words = [w for w in words if w not in used and w != target_word 
                                 and w not in wrong_options]
                    if len(other_words) >= 3 - len(related):
                        wrong_options.extend(random.sample(other_words, 3 - len(related)))
                    else:
                        break
                
                # Add correct answer at random position
                opts = wrong_options
                insert_pos = random.randint(0, 3)
                opts.insert(insert_pos, target_word)
                used.update(opts)
            
            if len(opts) == 4:
                # Build question with natural question phrasing
                q = (
                    f"Q{i+1}: {templates[i]}\n"
                    f"A) {opts[0].capitalize()} B) {opts[1].capitalize()} "
                    f"C) {opts[2].capitalize()} D) {opts[3].capitalize()}"
                )
                out.append(q)
        
        return "\n\n".join(out)
    try:
        prompt = (
            f"Generate {num_questions} multiple-choice questions with 4 options each (A-D) about the following text. "
            "For each question include the correct answer letter on the line 'Answer: <letter>'.\n\n"
            f"Text:\n{text}\n\nQuestions:"
        )
        try:
            resp = client.text2text_generation(prompt, model=TEXT2TEXT_MODEL, max_new_tokens=400, temperature=0.6)
        except Exception:
            resp = client.text_generation(prompt, model=FALLBACK_GEN_MODEL, max_new_tokens=400, temperature=0.6)
        out_text = ""
        if isinstance(resp, str):
            out_text = resp
        elif isinstance(resp, (list, tuple)) and resp:
            first = resp[0]
            out_text = (first.get("generated_text") if isinstance(first, dict) else str(first))
        elif hasattr(resp, "generated_text"):
            out_text = getattr(resp, "generated_text")
        else:
            out_text = str(resp)
        return out_text.strip()
    except Exception as e:
        
        return generate_quiz_with_hf(None, text, num_questions)

# ===== Main Execution =====
def main():
    """Main program execution flow."""
    # Get input file path
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter text file name (.txt): ").strip()
    
    # Read input text
    try:
        text = read_text(file_path)
    except Exception as e:
        print(e)
        return
    


    # Initialize API client
    client = init_hf()

    # Generate and display text summary
    print("\n--- Text Summarization ---")
    summary = summarize_with_hf(client, text)
    print(summary)

    # Get and validate number of keywords to generate
    def get_valid_keyword_count() -> int:
        while True:
            try:
                if len(sys.argv) >= 3:
                    count = int(sys.argv[2])
                else:
                    count = int(input("\nHow many keywords to generate? (1-10): "))
                
                if count < 1:
                    print("Error: Number must be at least 1. Please try again.")
                    if len(sys.argv) >= 3:  # If invalid arg provided, exit
                        return 0
                    continue
                elif count > 10:
                    print("Error: Number cannot exceed 10. Using maximum value of 10.")
                    return 10
                return count
                
            except ValueError:
                print("Error: Please enter a valid number between 1 and 10")
                if len(sys.argv) >= 3:  # If invalid arg provided, exit
                    return 0
                continue
    
    kcount = get_valid_keyword_count()
    if kcount == 0:  # Invalid command line argument
        return

    # Generate and display keywords
    print("\n--- Keyword Generation ---")
    keywords = generate_keywords_with_hf(client, text, kcount)
    print(", ".join(keywords))

    # Generate and display quiz
    print("\n--- Quiz Generation ---")
    quiz = generate_quiz_with_hf(client, text, num_questions=3)
    print(quiz)

if __name__ == "__main__":
    main()
