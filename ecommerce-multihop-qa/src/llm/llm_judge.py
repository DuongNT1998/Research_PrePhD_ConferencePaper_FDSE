import openai


class LLMJudge:
    def score_answer(self, query, answer, evidence):
        prompt = f"""
        Question: {query}
        Answer: {answer}
        Evidence: {evidence}

        Rate correctness from 0 to 1.
        """

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )

        return float(response["choices"][0]["message"]["content"])