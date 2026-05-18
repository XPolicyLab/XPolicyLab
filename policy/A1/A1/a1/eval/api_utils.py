import time

try:
    import openai
except ImportError:
    # Allow dependency to be optional
    openai = None


def get_chat_response(
    prompt,
    api_key,
    model="gpt-4-0613",
    temperature=0,
    max_tokens=256,
    n=1,
    patience=10000000,
    sleep_time=0
):
    """Run a query through an OpenAI model"""

    messages = [
        {"role": "user", "content": prompt},
    ]
    client = openai.OpenAI(
        api_key=api_key,
    )
    while patience > 0:
        patience -= 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction != None:
                    return prediction
            else:
                prediction = [choice.message.content.strip() for choice in response.choices]
                if prediction[0] != "" and prediction[0] != None:
                    return prediction

        except Exception as e:
            if "Rate limit" not in str(e):
                print(e)

            if "Please reduce the length of the messages" in str(e):
                print("!!Reduce prompt size")
                # reduce input prompt and keep the tail
                new_size = int(len(prompt) * 0.9)
                new_start = len(prompt) - new_size
                prompt = prompt[new_start:]
                messages = [
                    {"role": "user", "content": prompt},
                ]

            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""

