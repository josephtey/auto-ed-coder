def get_prompt_by_type(positive_samples, negative_samples, prompt_type):
    if prompt_type == "spam_classifier":
        prompt = f"""
        You are analyzing features learned by a neural network trained for spam classification. 
        Examine the text samples below and identify patterns that could be indicative or spam, or non-spam messages.
        
        Messages that strongly activate this feature:
        <positive-samples>{', '.join([sample.text for sample in positive_samples])}</positive-samples>
        
        Messages that don't activate this feature:
        <negative-samples>{', '.join([sample.text for sample in negative_samples])}</negative-samples>

        Before you respond, try to identify the most specific attribute that are shared by most positive samples, and none of the negative samples. 
        The more specific the better, as long most positive samples share that same attribute.

        First, describe your reasoning in a 'reasoning' field in JSON format.
        Then, describe the specific attributes in an 'attributes' field in JSON format.
        Finally, within a 'label' field in JSON format, write a concise 4-8 word label that captures the spam-related pattern.

        Output:
        {{
          "reasoning": "<string of your reasoning>",
          "attributes": "<string of attributes, comma delimited>",
          "label": "<string of label>"
        }}
        Output:"""
    else:
        # Default prompt
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

def get_score_prompt_by_type(prompt_type, samples, attributes):
    # Default scoring prompt
    prompt = f"""
    I have a dataset of text snippets:
    <samples>{', '.join([sample.text for sample in samples])}</samples>

    What percent of the samples fit the attribute described below?
    <attribute>{attributes}</attribute>

    Give your answer as an integer between 0 and 100 inclusive, in JSON format as follows:
    Do not include the percent sign in your response; include only the number itself.
    
    Output: {{
      "percent": <number>
    }}
    Output: """
    
    return prompt

# For backward compatibility
def label_prompt(positive_samples, negative_samples):
    return get_prompt_by_type("default", positive_samples, negative_samples)

def score_prompt(samples, attributes):
    return get_score_prompt_by_type("default", samples, attributes)
