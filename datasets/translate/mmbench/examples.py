examples = {
    "es":[
        {"role": "system",
            "content": """You are a helpful AI assistant that specializes in English to Spanish translations.
            Your task is to translate instruction datasets from English to Spanish.
            Here are some important guidelines:
            1. Maintain the original meaning and intent of the instructions
            2. Use standard Spanish language
            3. Keep the technical terms that don't have widely accepted Spanish translations
            4. Preserve the Question and Answer tags. Do not translate them.

            Please provide accurate Spanish translations for all text fields.
            """
        },
        {
            "role": "user",
            "content": f"Translate the following multiple choice questions and answers to Spanish \n\n\n Question: Each of the following situations relates to a different company. <image 1> For company B, find the missing amounts. \n Answer: 1.$63,020, 2.$58,410, 3.$71,320, 4.$77,490"
        },
        {
            "role": "assistant",
            "content": f"Question: Cada una de estas situaciones se refiere a una compañía diferente. <image 1> Para la compañía B, calcula la cantidad que falta. \n Answer: 1.$63,020, 2.$58,410, 3.$71,320, 4.$77,490"
        },
        {
            "role": "user",
            "content": f"Translate the following question and answers to Spanish \n\n\n Question: Solve for the unknown number of years in each of the following:<image 1> \n Answer: 1.10.52 years, 2.14.73 years, 3.20.02 years, 4.24.73 years"
        },
        {
            "role": "assistant",
            "content": f"Question: Calcula para el incierto número de años para cada uno de estos: <⁣image 1> \n Answer: 1.10.42 años, 2.14.73 años, 3.20.02 años, 4.24.73 años"
        }
    ],
    "eu":[
        {"role": "system",
            "content": """You are a helpful AI assistant that specializes in English to Basque translations.
            Your task is to translate instruction datasets from English to Basque.
            Here are some important guidelines:
            1. Maintain the original meaning and intent of the instructions
            2. Use standard Basque language (batua)
            3. Keep the technical terms that don't have widely accepted Basque translations
            4. Preserve the Question and Answer tags. Do not translate them.

            Please provide accurate Basque translations for all text fields.
            """
        },
        {
            "role": "user",
            "content": f"Translate the following multiple choice questions and answers to Basque \n\n\n Question: Each of the following situations relates to a different company. <image 1> For company B, find the missing amounts. \n Answer: 1.$63,020, 2.$58,410, 3.$71,320, 4.$77,490"
        },
        {
            "role": "assistant",
            "content": f"Question: Hurrengo situazio hauetako bakoitza konpainia desberdin baten inguruan da. <image 1> B konpainiareko bilatu falta den kanitatea, \n Answer: 1.$63,020, 2.$58,410, 3.$71,320, 4.$77,490"
        },
        {
            "role": "user",
            "content": f"Translate the following question and answers to Basque \n\n\n Question: Solve for the unknown number of years in each of the following:<image 1> \n Answer: 1.10.52 years, 2.14.73 years, 3.20.02 years, 4.24.73 years"

        },
        {
            "role": "assistant",
            "content": f"Question: Kalkulatu zenbat urte faltako diren hurrengo irudian:<image 1> \n Answer: 1.10.42 urte, 2.14.73 urte, 3.20.02 urte, 4.24.73 urte"
        }
    ],

    "cat":[
        {"role": "system",
            "content": """You are a helpful AI assistant that specializes in English to Catalan translations.
            Your task is to translate instruction datasets from English to Catalan.
            Here are some important guidelines:
            1. Maintain the original meaning and intent of the instructions
            2. Use standard Catalan language
            3. Keep the technical terms that don't have widely accepted Catalan translations
            4. Preserve the Question and Answer tags. Do not translate them.
            Please provide accurate Catalan translations for all text fields.
            """
        },
        {
            "role": "user",
            "content": f"Translate the following multiple choice questions and answers to Catalan \n\n\n Question: Each of the following situations relates to a different company. <image 1> For company B, find the missing amounts. \n Answer: 1.$63,020, 2.$58,410, 3.$71,320, 4.$77,490"
        },
        {
            "role": "assistant",
            "content": f"Question: Cada una de les situacions següents fa referència a una empresa diferent. <image 1> Per a l'empresa B, troba les quantitats que falten, \n Answer: 1.$63,020, 2.$58,410, 3.$71,320, 4.$77,490"
        },
        {
            "role": "user",
            "content": f"Translate the following question and answers to Catalan \n\n\n Question: Solve for the unknown number of years in each of the following:<image 1> \n Answer: 1.10.52 years, 2.14.73 years, 3.20.02 years, 4.24.73 years"
        },
        {
            "role": "assistant",
            "content": f"Question: Resol el nombre desconegut d'anys per a cadascun dels següents casos:<image 1> \n Answer: 1.10.42 anys, 2.14.73 anys, 3.20.02 anys, 4.24.73 anys"
        }
    ],
    "gal": [
        {"role": "system",
        "content": """You are a helpful AI assistant that specializes in English to Galician translations.
        Your task is to translate instruction datasets from English to Galician.
        Here are some important guidelines:
        1. Maintain the original meaning and intent of the instructions
        2. Use standard Galician language
        3. Keep the technical terms that don't have widely accepted Galician translations
        4. Preserve the Question and Answer tags. Do not translate them.
        Please provide accurate Galician translations for all text fields.
        """
        },
        {
            "role": "user",
            "content": f"Translate the following multiple choice questions and answers to Galician \n\n\n Question: Each of the following situations relates to a different company. <image 1> For company B, find the missing amounts. \n Answer: 1.$63,020, 2.$58,410, 3.$71,320, 4.$77,490"
        },
        {
            "role": "assistant",
            "content": f"Question:Cada unha destas situacións refírese a unha compañía diferente. <image 1> Para a compañía B, calcula a cantidade que falta. \n Answer: 1.$63,020, 2.$58,410, 3.$71,320, 4.$77,490"
        },
        {
            "role": "user",
            "content": f"Translate the following question and answers to Galician \n\n\n Question: Solve for the unknown number of years in each of the following:<image 1> \n Answer: 1.10.52 years, 2.14.73 years, 3.20.02 years, 4.24.73 years"
        },
        {
            "role": "assistant",
            "content": f"Question: Calcula para o incerto número de anos para cada un destes: <image 1> \n Answer: 1.10.42 anos, 2.14.73 anos, 3.20.02 anos, 4.24.73 anos"
        }
    ]
}