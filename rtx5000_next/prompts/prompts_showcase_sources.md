# Sources of prompts_showcase.json

First 27 prompts are verbatim examples published by the model authors (collected by web search, 2026-11-27), in file order:
- PixArt-Sigma (17): https://pixart-alpha.github.io/PixArt-sigma-project/ , https://github.com/PixArt-alpha/PixArt-sigma (README and asset/samples.txt), https://huggingface.co/spaces/PixArt-alpha/PixArt-Sigma
- SD3.5 (3): https://huggingface.co/stabilityai/stable-diffusion-3.5-large and its Space (the "Astronaut in a jungle" and capybara sign prompts)
- FLUX.1-schnell (3): https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell
- Sana (4, English only): https://github.com/NVlabs/Sana (asset/samples/samples_mini.txt, app/app_sana.py), https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers
- "portrait photo of a girl ..." (Sana app default).
The last two prompts (fisherman, ramen with text) come from our own earlier set `prompts.json`, kept for continuity with Eval 2.
Chinese and emoji Sana prompts were dropped: the PixArt T5 text encoder does not handle them well.
Bias note: these are author-chosen examples, so they favour what each model does well. The report must say this. Licence: no licence statement found for the prompt text itself.
The collector truncated a few long prompts (e.g. "covered in."); the puppies prompt is therefore cut short and should be verified against the Space before quoting it.
