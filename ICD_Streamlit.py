import streamlit as st
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Load the model and tokenizer
@st.cache_resource()
def load_model():
    model_name = "edvenswa/ICD-10-Codes-7-2epochs"
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=None  # Add HUGGINGFACE_TOKEN if needed
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

model, tokenizer = load_model()

# Custom prompt style (User can change this)
def get_prompt(disease_name, instruction):
    return f"""{instruction}

### Input:
{disease_name}

### Output:
"""

# Streamlit UI
st.title("ICD-10 Code Finder")
st.markdown("Enter disease names to find their corresponding **ICD-10** codes.")

# User input fields
instruction = st.text_area(
    "Custom Instruction (Optional)",
    "You are a medical expert with advanced knowledge in clinical codes for treatment planning."
)
disease_input = st.text_area("Enter disease names (one per line)")

if st.button("Get ICD Codes"):
    if disease_input.strip():
        diseases = disease_input.split("\n")
        icd_codes = {}

        for disease in diseases:
            prompt = get_prompt(disease.strip(), instruction)
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

            # Generate output
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,  # Reduce token limit for efficiency
                use_cache=True,
            )

            # Decode output
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            icd_code = response.split("### Output:")[-1].strip()  # Extract the ICD code
            icd_codes[disease.strip()] = icd_code

        # Display results
        st.subheader("Results")
        for disease, icd_code in icd_codes.items():
            st.write(f"**{disease}** → `{icd_code}`")

    else:
        st.warning("Please enter at least one disease name.")

# Footer
st.markdown("---")
st.markdown("🚀 Built with **Streamlit** & **Unsloth** | [GitHub](https://github.com/)")

