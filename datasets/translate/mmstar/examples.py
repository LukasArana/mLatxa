examples ={
        "eu":[
            {"role": "system",
                "content": """You are a helpful AI assistant that specializes in English to Basque translations.
                Your task is to translate instruction datasets from English to Basque.

                Here are some important guidelines:
                1. Maintain the original meaning and intent of the instructions
                2. Use standard Basque language (batua)
                3. Keep the technical terms that don't have widely accepted Basque translations
                4. Strictly preserve the format of the instructions. Do not translate them.
                5. Never use parenthesis in the option choices. (A) -> A, (B) -> B, (C) -> C, (D) -> D
                6. Translate these words as given: Hint -> Argibidea, Question -> Galdera, Options -> Aukerak 

                Please provide accurate Basque translations for all text fields."""
            },
            {
                "role": "user",
                "content": f"Translate the following example to Basque \n\n\n What feeling is shown in this image?\nOptions: A: engaged, B: distressed, C: angry, D: love"
            },
            {
                "role": "assistant",
                "content": f"Nola daude irudiko pertsonak? Aukerak: A: inplikatuta, B: larrituta, C: haserre,  D: maitekor"
            },
            {
                "role": "user",
                "content": f"Translate the following example to Basque \n\n\n What is the overall theme of this image?\nOptions: (A): Sports, (B): Sunset, (C): Nature, (D): Kids playing"

            },
            {
                "role": "assistant",
                "content": f"Zein da irudiaren gai nagusia? Aukerak: A: kirola, B: ilunabarra, C: natura,  D: haurrak jolasean"
            }
        ]
    }