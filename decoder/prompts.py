def label_prompt_with_token_scores(positive_samples, negative_samples):
    prompt = f"""
    What are the attributes that texts within <positive-samples></positive-samples> share, that are not shared by those in <negative-samples></negative-samples>?
    <positive-samples>{', '.join([sample.text for sample in positive_samples])}</positive-samples>
    <negative-samples>{', '.join([sample.text for sample in negative_samples])}</negative-samples>
    Before you respond, try to identify the most specific attribute that are shared by all positive samples,
    and none of the negative samples. The more specific the better, as long as only all positive samples share that same attribute.

    First, describe your reasoning in a 'reasoning' field in JSON format.
    Then, describe the shared attributes in an 'attributes' field in JSON format. Within this field, do not reference any words as 'positive' or 'negative'; simply refer to samples as 'the samples'.
    Finally, within a 'label' field in JSON format, write a short, 4-8 word label for this attribute in sentence case.
    Output:
    {{
      "reasoning": "<string of your reasoning>",
      "attributes": "<string of attributes, comma delimited>",
      "label": "<string of label>"
    }}
    Output:"""
    return prompt