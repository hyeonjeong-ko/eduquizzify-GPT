# prompts.py
from langchain.prompts import ChatPromptTemplate

# 객관식 퀴즈 및 해설 추가 Prompt
questions_prompt_with_explanation = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.
            Based ONLY on the following context make 5 (FIVE) questions to test the user's knowledge about the text.
            Each question should have 5 answers, four of them must be incorrect and one should be correct.
            After each question, provide a brief explanation to clarify the correct answer.
            Use (o) to signal the correct answer.
            Context: {context}
            """,
        )
    ]
)

# JSON 포맷으로 변환하는 Prompt (해설 포함)
formatting_prompt_with_explanation = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a powerful formatting algorithm.
            You format exam questions with explanations into JSON format.
            Answers with (o) are the correct ones.
            Example Input:
            Question: What is the color of the ocean?
            Answers: Red|Yellow|Green|Blue(o)
            Explanation: The ocean appears blue because water absorbs colors in the red part of the light spectrum.
            Example Output:
            ```json
            {{ "questions": [
                    {{
                        "question": "What is the color of the ocean?",
                        "answers": [
                                {{ "answer": "Red", "correct": false }},
                                {{ "answer": "Yellow", "correct": false }},
                                {{ "answer": "Green", "correct": false }},
                                {{ "answer": "Blue", "correct": true }}
                        ],
                        "explanation": "The ocean appears blue because water absorbs colors in the red part of the light spectrum."
                    }}
                ]
            }}
            ```
            Your turn!
            Questions: {context}
            """,
        )
    ]
)

# Short-Answer Quiz Prompt
short_question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that is role playing as a teacher.
            
            Based ONLY on the following context, generate 5 (FIVE) short-answer questions to test the user's knowledge about the text.
            
            Each question should be open-ended and require the user to provide a brief answer.
            
            Question example:
            
            Question: What is the color of the ocean?
            Answer: Blue
             
            Your turn!
             
            Context: {context}
            """,
        )
    ]
)

short_formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a powerful formatting algorithm.
             
            You format short-answer quiz questions into JSON format.
             
            Example Input:

            Question: What is the color of the ocean?
            Answer: Blue
                 
            Question: What is the capital of Georgia?
            Answer: Tbilisi
             
            Example Output:
             
            ```json
            {{ "questions": [
                    {{
                        "question": "What is the color of the ocean?",
                        "answer": "Blue"
                    }},
                    {{
                        "question": "What is the capital of Georgia?",
                        "answer": "Tbilisi"
                    }}
                ]
             }}
            ```
            Your turn!

            Questions: {context}
            """,
        )
    ]
)

# 학습 카드 생성 Prompt
learning_cards_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that acts as a teacher.

            Based ONLY on the following context, create 5 (FIVE) learning cards to help the user study the material.

            Each learning card should contain a key fact or concept derived from the text.

            Each card should have three components:
              - A statement of the key fact or concept
              - A question related to the fact or concept in parentheses
              - A brief explanation that elaborates or clarifies the key fact

            Format the card in the following structure:

            Statement: [The key fact]
            Question: [A relevant question]
            Explanation: [A brief explanation of the fact or concept]

            Separate each learning card with a newline character.

            Context: {context}
            """,
        )
    ]
)

# 학습 카드 포매터 Prompt
learning_cards_formatter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a formatting assistant.

            Ensure that the following learning cards are properly formatted as a JSON object with a key "learning_cards" that contains an array of objects.
            Each object should have three fields:
              - "statement" (the key fact or concept)
              - "question" (a relevant question about the fact)
              - "explanation" (a brief explanation or clarification of the fact)

            Format the JSON object exactly in the following way:

            Example Input:
            Statement: Ahn Jung-geun assassinated Japanese Resident General Ito Hirobumi at Harbin Station in 1909.
            Question: Who did Ahn Jung-geun assassinate at Harbin Station in 1909?
            Explanation: Ahn Jung-geun is considered a national hero for his actions, which were part of the Korean independence movement.

            Example Output:
            ```json
            {{
                "learning_cards": [
                    {{
                        "statement": "Ahn Jung-geun assassinated Japanese Resident General Ito Hirobumi at Harbin Station in 1909.",
                        "question": "Who did Ahn Jung-geun assassinate at Harbin Station in 1909?",
                        "explanation": "Ahn Jung-geun is considered a national hero for his actions, which were part of the Korean independence movement."
                    }}
                ]
            }}
            ```

            Ensure the output is formatted as shown.

            Learning Cards: {context}
            """,
        )
    ]
)
