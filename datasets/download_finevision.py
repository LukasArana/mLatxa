import os
import pandas as pd

subsets = ['CoSyn_400k_chart', 'CoSyn_400k_chemical', 'CoSyn_400k_circuit', 'CoSyn_400k_diagram', 'CoSyn_400k_document',
 'CoSyn_400k_graphic', 'CoSyn_400k_math', 'CoSyn_400k_music', 'CoSyn_400k_nutrition', 'CoSyn_400k_table',
 'DoclingMatix', 'LLaVA_Instruct_150K', 'SynthChartNet', 'SynthCodeNet', 'SynthFormulaNet', 'Unichart',
 'a_okvqa', 'aguvis-stage-1', 'ai2d_merged', 'alfworldgpt', 'allava_laion', 'allava_vflan', 'aokvqa',
 'art', 'arxivqa', 'bentham', 'blockdiagramcomputerized', 'blockdiagramhandwritten',
 'cambrian(filtered)_processed', 'captcha', 'chart2text', 'chartqa', 'chinesememe', 'chrome_writting',
 'clevr', 'clevr_math', 'clevr_math(mathv360k)', 'coco_colors', 'cocoqa', 'cocotext', 'ctw', 'datik',
 'datikz', 'densefusion_1m', 'diagram_image_to_text', 'docvqa', 'drivelm', 'dvqa', 'est_vqa', 'face_emotion',
 'figureqa', 'figureqa(mathv360k)', 'finqa', 'funsd', 'geo170k(align)', 'geo170k(qa)', 'geo3k',
 'geometry3k(mathv360k)', 'geomverse', 'geoqa+(mathv360k)', 'geos(mathv360k)', 'google_landmarks',
 'groundui', 'handwriting_forms', 'hateful_memes', 'hitab', 'hme100k', 'hw_squad', 'iam', 'iconqa',
 'iconqa(mathv360k)', 'idk', 'iiit5k', 'image_textualization(filtered)', 'imgur5k', 'indoor_qa',
 'infographic(gpt4v)', 'infographic_vqa', 'infographic_vqa_llava_format', 'intergps', 'invoices_receipts',
 'k12_printing', 'laion_gpt4v', 'latex_handwritten', 'latexformulas', 'llavar_gpt4_20k', 'lnqa',
 'localized_narratives', 'lrv_chart', 'lrv_normal(filtered)', 'lvis_instruct4v', 'mapqa', 'mapqa(mathv360k)',
 'maptext', 'mathwriting-google', 'mavis_math_metagen', 'mavis_math_rule_geo', 'memotion', 'mimic_cgd',
 'mmc_instruct', 'mmevol', 'mmra', 'mmsoc_memotion', 'multihiertt', 'nlvr2', 'objects365_qa', 'ocrvqa',
 'olmOCR-mix-0225-books', 'olmOCR-mix-0225-documents', 'oodvqa', 'orand_car_a', 'pathvqa', 'pdfvqa', 'plotqa',
 'pmc_vqa(mathv360k)', 'raven', 'rendered_text', 'robut_sqa', 'robut_wikisql', 'robut_wtq', 'scienceqa',
 'scienceqa(nona_context)', 'screen2words', 'screenqa', 'sharegpt4o', 'sharegpt4v(coco)',
 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 'sketchyvqa', 'slidevqa',
 'spark', 'spatialsense', 'spot_the_diff', 'sroie', 'st_vqa', 'sujet_finance', 'super_clevr(mathv360k)',
 'svrd', 'synthdog', 'tabmwp', 'tabmwp(mathv360k)', 'tal_ocr_eng', 'tallyqa', 'tat_dqa', 'tat_qa',
 'text_OpenMathInstruct-2', 'text_code_feedback', 'text_codefeedback_filtered_instruction',
 'text_infinitymath', 'text_mathinstruct', 'text_mathqa', 'text_mathstepdpo10k', 'text_numinamath_cot',
 'text_openhermes_2_5', 'text_openorca', 'text_orcamath', 'text_pythoncode25k', 'text_pythoncodealpaca',
 'text_ruozhiba', 'text_theoremqa', 'text_wizardlm_evol', 'textcaps', 'textocr(gpt4v)', 'textvqa', 'tqa',
 'unigeo(mathv360k)', 'ureader_cap', 'ureader_ie', 'ureader_kg_processed', 'ureader_qa_processed',
 'vision_flan(filtered)', 'vistext', 'visual7w', 'visualmrc', 'visualwebinstruct(filtered)',
 'vizwiz(mathv360k)', 'vqaonbd', 'vqarad', 'vqav2', 'vsr', 'websight', 'wildvision', 'wordart', 'yesbut']

from datasets import load_dataset, get_dataset_config_names

# Get all config names
dataset_name = "HuggingFaceM4/FineVision"  # Replace with actual dataset name
# Download and save each config
base_path = "/leonardo_scratch/fast/AIFAC_5C0_261/datasets/train"
rows = {}
for subset in subsets:
    print(f"Downloading subset: {subset}")
    dataset = load_dataset(dataset_name, subset, num_proc = 8, cache_dir=os.path.join(base_path, subset))["train"]
    rows[subset] = len(dataset)

print(rows)
pd.DataFrame(list(rows.items()), columns=['subset', 'num_rows']).to_csv('subset_names.csv', index=False)
