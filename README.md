#  GenAI Content Generation Pipeline

A beginner-friendly pipeline to **generate stickers, phrases, and pop-culture references** based on the selected country, using **AI models like GPT, Gemini, FLUX, and BRIA**.

Whether you're creating content for India or Italy, this project helps you automate the full flow from **prompt generation** to **image creation** to **styling** — all with a single command.

---

##  Project Components

```
├── cleaned_flux.py               # Main script to generate styled sticker images
├── gpt_test.py                   # Generates prompts using GPT (expression-based)
├── gpt_categories.py             # Generates prompts using GPT (category-based)
├── gemini_test.py                # Uses Gemini API for prompt generation
├── setup_model.py                # Loads the FLUX image model and LLMs
├── helper_function.py            # Utility functions: text cleaning, masking, etc.
├── automatic.sh                  # Shell script to automate the whole pipeline
├── keyboardPopTextStyles.json    # Styles database (fonts, bg images, keyword logic)
├── pop_text_font.json            # Category-to-font style mappings
├── color_map2.json               # Category-to-color mapping
```

---

##  How to Use

### Step 1: Run the entire pipeline using one command

```bash
bash automatic.sh <country_name>
```

Example:

```bash
bash automatic.sh india
```

This will:
- Generate prompts
- Generate images using Flux
- Apply backgrounds and text styles
- Save final outputs inside:
```
/genAIcontentgeneration-assets/outputs/
```

---

##  Key Features

-  **Prompt Generation**  
  Uses GPT-4 or Gemini to generate fun, culture-specific prompts for stickers.

-  **Image Generation with Style**  
  FLUX model creates human-centric expressions with BRIA removing image backgrounds.

-  **Multilingual & Cultural Support**  
  Generate outputs specific to local culture, language, and context.

-  **Style + Color Mapping**  
  Customizes font, background, and colors per category using JSON mappings.

-  **Personalized / Non-Personalized Stickers**  
  Supports both general and user-specific content generation modes.

---

##  Requirements

- Python 3.8+
- CUDA-enabled GPU

### Install Python Dependencies

```bash
pip install torch diffusers transformers rembg briarmbg numpy opencv-python
```

Or use:

```bash
pip install -r requirements.txt
```

---

##  Run the Image Pipeline with Optional Flags

To run with API (like Gemini), use:

```bash
bash automatic.sh <country_name>
```

The `automatic.sh` might look like this:

```bash
CUDA_VISIBLE_DEVICES=1 python /genAIcontentgeneration/cleaned_flux.py --country "$country_name" --json1 "$json_path" --output "$image_output_path" --json2 "$json_path_categories" --sticker_type "$sticker_type" --font "$font" --api
```

---

##  Configuration Files Explained

| File                     | Purpose                                             |
|--------------------------|-----------------------------------------------------|
| `keyboardPopTextStyles.json` | Defines how stickers look: font, bg, keywords     |
| `pop_text_font.json`     | Maps each category to specific text style IDs      |
| `color_map2.json`        | Assigns unique color sets to different categories  |

---

##  AI Models Used

###  Language Models:
- **GPT-4** via OpenAI (in `gpt_test.py`, `gpt_categories.py`)
- **Gemini** via API (in `gemini_test.py`)
- **Meta LLaMA 3** (optional via `setup_model.py`)

###  Image Models:
- **FLUX.1 by Black Forest Labs** – Generates stylized characters
- **BRIA RMBG** – Removes background from generated images
- **LoRA Expression Models** – Optional, for better facial emotion control

---

##  Prompt Design Logic

Each generated prompt is:
- Human-centric (no animals or mascots)
- Localized to your chosen country
- Emotionally expressive (poses, feelings, actions)
- Max 40 words
- Enriched with keywords, outfit suggestions, and props

---

##  Perfect For

- Junior developers exploring content pipelines
- Designers automating sticker creation
- LLM + Diffusion model enthusiasts
- Anyone building culturally-aware GenAI systems

---


