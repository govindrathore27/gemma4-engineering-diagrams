---
license: cc-by-4.0
task_categories:
- image-classification
- visual-question-answering
- image-to-text
language:
- en
tags:
- engineering
- technical-drawings
- as1100
- compliance
- computer-vision
- multimodal
size_categories:
- 100<n<1K
dataset_info:
  features:
  - name: image
    dtype: image
  - name: messages_json
    dtype: string
  - name: part_number
    dtype: string
  - name: error_code
    dtype: string
  - name: description
    dtype: string
  - name: filename
    dtype: string
  - name: system_prompt
    dtype: string
  - name: user_query
    dtype: string
  - name: assistant_response
    dtype: string
  splits:
  - name: train
    num_bytes: 10131224.0
    num_examples: 19
  - name: validation
    num_bytes: 1060420.0
    num_examples: 2
  - name: test
    num_bytes: 1580424.0
    num_examples: 3
  download_size: 9058894
  dataset_size: 12772068.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# Engineering Drawings AS1100 Compliance Dataset

## Dataset Description

This dataset contains engineering drawings with various AS1100 (Australian Standard for Technical Drawing) compliance issues for training AI models to identify missing elements and non-compliance issues in technical drawings.

### Dataset Summary

The Engineering Drawings AS1100 Compliance Dataset is designed to train and evaluate vision-language models on identifying compliance issues in technical engineering drawings according to the AS1100 Australian standard. The dataset includes drawings with systematically introduced faults in title blocks, orthographic views, and drawing elements.

- **Repository:** [Your GitHub Repository]
- **Paper:** [Your Thesis/Paper Link if available]
- **Point of Contact:** [Your Email]

### Supported Tasks

- **Visual Question Answering**: Models can answer questions about what elements are missing from engineering drawings
- **Image Classification**: Classify drawings as compliant or non-compliant with AS1100 standards
- **Technical Drawing Analysis**: Identify specific missing elements in title blocks and views
- **Multi-modal Conversation**: Engage in conversations about technical drawing compliance

### Languages

- English (en)

## Dataset Structure

### Data Instances

Each data instance contains:
- An engineering drawing image (JPG format)
- A conversation structure with system prompt, user query, and assistant response
- Metadata including part number, error code, and description

Example:
```json
{
  "image": "<PIL.Image>",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert engineering drawing analyst specializing in AS1100 Australian standard compliance."
    },
    {
      "role": "user", 
      "content": [
        {
          "type": "text",
          "text": "Please analyze this engineering drawing for AS1100 compliance. Identify any missing elements in the title block or views."
        },
        {
          "type": "image_url",
          "image_url": {"url": "<image>"}
        }
      ]
    },
    {
      "role": "assistant",
      "content": "Missing title in the title block."
    }
  ],
  "part_number": "Part01",
  "error_code": "TB01",
  "description": "MissingTitle",
  "filename": "Part01_TB01_MissingTitle.jpg"
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `image` | Image | Engineering drawing image in JPG format |
| `messages` | List[Dict] | Conversation format with system, user, and assistant messages |
| `part_number` | String | Identifier for the engineering part (e.g., "Part01") |
| `error_code` | String | Code identifying the type of compliance issue (e.g., "TB01", "PERFECT") |
| `description` | String | Brief description of the missing element |
| `filename` | String | Original filename of the image |

### Data Splits

| Split | Examples |
|-------|----------|
| train | 168 |
| validation | 21 |
| test | 21 |

**Total:** 210 examples across 5 different engineering parts

## Dataset Creation

### Curation Rationale

This dataset was created to address the need for automated compliance checking of engineering drawings according to AS1100 standards. Manual review of technical drawings is time-consuming and error-prone, making automated analysis valuable for:

- Quality assurance in engineering firms
- Educational tools for technical drawing courses
- Standardization compliance in manufacturing
- Research in computer vision for technical documents

### Source Data

#### Initial Data Collection and Normalization

The base engineering drawings represent common mechanical components with standard orthographic projections including:
- Front view (elevation)
- Top view (plan)
- Right side view
- Isometric view (where applicable)

Each drawing includes a complete AS1100-compliant title block with:
- Drawing title
- Material specification (Mild Steel)
- Scale (1:1)
- General tolerance (±0.1)
- Surface finish (Ra 3.2)
- Drawn by field
- Date
- Drawing number

#### Who are the source language producers?

The drawings and annotations were created by engineering researchers familiar with AS1100 standards and technical drawing conventions.

### Annotations

#### Annotation process

Systematic fault injection was performed following a structured approach:

1. **Single Faults**: Individual elements removed (16 variations per part)
2. **Double Faults**: Two elements removed simultaneously (15 combinations per part)  
3. **Triple Faults**: Three elements removed (10 combinations per part)
4. **Perfect Baseline**: Complete compliant drawing (1 per part)

#### Who are the annotators?

Annotations were created by the dataset authors with expertise in:
- AS1100 Australian technical drawing standards
- Engineering drawing conventions
- CAD software and technical documentation

### Personal and Sensitive Information

This dataset contains no personal or sensitive information. All drawings are of generic mechanical components with fictitious part numbers and standard technical specifications.

## Considerations for Using the Data

### Social Impact of Dataset

**Positive Impacts:**
- Improved quality control in engineering documentation
- Educational tool for teaching technical drawing standards
- Reduced manual review time for compliance checking
- Standardization of drawing review processes

**Potential Concerns:**
- Over-reliance on automated systems without human oversight
- Model may not generalize to drawing styles not represented in training data

### Discussion of Biases

The dataset has several inherent biases:

- **Drawing Style Bias**: Limited to a specific CAD software output style
- **Complexity Bias**: Focuses on relatively simple mechanical parts
- **Standard Bias**: Specifically targets AS1100 (Australian) standards
- **Language Bias**: Annotations and responses are in English only

### Other Known Limitations

- Limited to 2D orthographic projections
- Does not include assembly drawings or complex technical details
- Fault categories are predefined and may not cover all real-world compliance issues
- Image quality dependent on PDF-to-image conversion process

## Additional Information

### Dataset Curators

- [Your Name], [Your Institution]
- [Supervisor Name], [Institution] (if applicable)

### Licensing Information

This dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

### Citation Information

```bibtex
@misc{engineering_drawings_as1100_2025,
  title={Engineering Drawings AS1100 Compliance Dataset},
  author={[Your Name]},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/jcrzd/engineering-drawings-as1100}}
}
```

### Contributions

Thanks to the following contributors:
- [Your Name] - Dataset creation and curation
- [Supervisor/Collaborators] - Domain expertise and validation

### Changelog

- **v1.0.0** (2025-07-25): Initial release with 210 annotated engineering drawings across 5 parts

### Contact

For questions about this dataset, please contact [your.email@university.edu] or open an issue in the [dataset repository].

---

## Usage Examples

### Loading the Dataset

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("jcrzd/engineering-drawings-as1100")

# Load specific split
train_dataset = load_dataset("jcrzd/engineering-drawings-as1100", split="train")
```

### Basic Usage

```python
import matplotlib.pyplot as plt

# Display first example
example = dataset['train'][0]
plt.imshow(example['image'])
plt.title(f"Part: {example['part_number']} | Error: {example['error_code']}")
plt.show()

print("Assistant Response:", example['messages'][2]['content'])
```

### Filter by Error Type

```python
# Get all title block errors
title_block_errors = dataset['train'].filter(lambda x: x['error_code'].startswith('TB'))

# Get perfect drawings (no errors)
perfect_drawings = dataset['train'].filter(lambda x: x['error_code'] == 'PERFECT')
```
