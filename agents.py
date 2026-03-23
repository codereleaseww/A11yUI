import json
import re
from typing import Iterable

SEED = 2026
BACKBONE = None  # f(prompt, text_imgs, temperature, seed, n)


def _extract_first_html_block(text: str):
    match = re.search(r"```html(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None


def _extract_html_blocks(items: Iterable[str]):
    if isinstance(items, str):
        block = _extract_first_html_block(items)
        return [block] if block else []

    blocks = []
    for item in items:
        block = _extract_first_html_block(item)
        if block:
            blocks.append(block)
    return blocks


def _parse_score_rows(text: str):
    rows = []
    cleaned = text.strip().strip("```").strip()
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values = [int(value.strip()) for value in line.split(",")]
        except ValueError:
            continue
        values.append(sum(values))
        rows.append(values)
    return rows


def _extract_json_block(text: str):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else None


def _parse_json_object(text: str):
    json_block = _extract_json_block(text)
    if json_block:
        try:
            parsed = json.loads(json_block)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {}


class MLLMAgent:
    def __init__(self, prompt):
        self.prompt = prompt
        self.do_sample = False

    def infer(self, textx_imgs, parse=True, temperature=0, seed=SEED, n=1):
        if BACKBONE is None:
            raise RuntimeError("BACKBONE is not configured. Set agents.BACKBONE before calling infer().")

        self.do_sample = n > 1
        text = BACKBONE(self.prompt, textx_imgs, temperature, seed, n)
        return self.parser(text) if parse else text
    
    def parser(self, text):
        return text


class HTMLBlockAgent(MLLMAgent):
    def parser(self, text):
        return _extract_first_html_block(text)


class SampledHTMLBlockAgent(MLLMAgent):
    def parser(self, text):
        if self.do_sample:
            return _extract_html_blocks(text)
        block = _extract_first_html_block(text)
        return block if block else text


class AgentGenerate(SampledHTMLBlockAgent):
    def __init__(self):
        super().__init__("""
You are an expert GUI developer.

Based on the reference screenshot of a specific section of a webpage (such as the header, footer, card, etc.) provided by the user, build a single-page app using HTML, CSS, and JS. Please follow the detailed requirements below to ensure the generated code is accurate:

### Basic Requirements:
                         
1. **Rigid Requirements**
   - You are provided with the following unmodifiable HTML framework:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <!-- Your task is to fill this area -->
</body>
</html>
```
   - Your task is to generate a code block that starts with a <div> tag and ends with a </div> tag, and embed it within the <body> tag of the above - mentioned framework.
   - Do not deliberately center the content. Arrange the elements according to their original layout and positions.
   - The generated code should not have fixed width and height settings.
   - Ensure that the proportions of images in the code are preserved.
   - Both the margin and padding in the code should be set to 0.
   - Make sure that the generated code does not conflict with outer <div> elements in terms of layout and style.
   - The final return should be the complete HTML code, that is, including the above - mentioned framework and the code you generated and embedded into the <body> of the framework.

2. **Appearance and Layout Consistency:**
   - Ensure the app looks exactly like the screenshot, including the position, hierarchy, and content of all elements.
   - The generated HTML elements and CSS classes should match those in the screenshot, ensuring that text, colors, fonts, padding, margins, borders, and other styles are perfectly aligned.

3. **Content Consistency:**
   - Use the exact text from the screenshot, ensuring the content of every element matches the image.
   - For images, use placeholder images from https://placehold.co and include a detailed description in the alt text for AI-generated images.

4. **No Comments or Placeholders:**
   - Do not add comments like "<!-- Add other navigation links as needed -->" or "<!-- ... other news items ... -->". Write the full, complete code for each element.

5. **Libraries to Use:**
   - Use the following libraries:
     - Google Fonts: Use the relevant fonts from the screenshot.
                         

### Process Steps:

1. **Analyze the Section:**
   Based on the provided screenshot, analyze a specific section of the webpage (such as the header, footer, card, form, etc.). Break down all the elements in this section (e.g., text, images, buttons, etc.) and understand their relative positions and hierarchy.

2. **Generate HTML Code:**
   Based on the analysis from Step 1, generate a complete HTML code snippet representing that specific section, ensuring all elements, positions, and styles match the screenshot.

3. **Text Content Comparison:**
   Compare the generated HTML with the screenshot’s text content to ensure accuracy. If there are any discrepancies or missing content, make corrections.

4. **Color Comparison:**
   Compare the text color and background color in the generated HTML with those in the screenshot. If they don't match, adjust the CSS classes and styles to reflect the correct colors.

5. **Background and Other Style Comparison:**
   Ensure the background colors, borders, padding, margins, and other styles in the generated HTML accurately reflect the design shown in the screenshot.

6. **Final Integration:**
   After reviewing and refining the previous steps, ensure that the generated HTML code is complete and perfectly matches the specific section of the screenshot.

### Code Format:

Please return the complete HTML code 
                         """)

class AgentAssemble(HTMLBlockAgent):
    def __init__(self):
        super().__init__("""
        You are an experienced front-end developer tasked with assembling multiple webpage module codes into a complete webpage.

        # CONTEXT #
        I will provide a screenshot of a webpage, the location information for each module, and the corresponding module code. 
        Your task is to assemble these modules into a complete webpage code based on their positions.

        # OBJECTIVE #
        Generate a complete HTML file that ensures the layout, style, and content of each module match the original webpage.

        # RESPONSE #
        You need to return the final assembled complete HTML code, for example:
        ```html
        code
        ```

        # steps #
        Please follow the steps below:

        **step1**: Analyze the webpage screenshot and the position information of each module.
        - Based on the screenshot and the module position data, understand the relative placement and layout of each module.
        - The position of each module is defined by two coordinates: the top-left corner [x1, y1] and the bottom-right corner [x2, y2]. x1 < x2 and y1 < y2. The coordinate values range from 0 to 1, representing the ratio relative to the width and height of the image.

        **step2**: Assemble the HTML code and CSS classes of each module based on its position.
        - Use the provided module code to stitch the modules together in the correct order and position.
        - Ensure that the modules do not overlap and that the layout is correct.

        **step3**: Review and fix the assembled webpage.
        - Compare the generated webpage with the screenshot to ensure the content, layout, and styles match exactly.
        - If any issues such as misalignment, overlapping, or missing content are found, fix them.
        - Review again and fix it until the generated webpage exactly matches the screenshot.

        **step4**: Generate the final HTML code.
        - Based on the checks and fixes from step 3, generate the final complete webpage code that exactly matches the screenshot.

        # Notes #
        - There should be no overlap between modules.
        - For image modules, use placeholder images from https://placehold.co and provide a detailed description in the alt text for AI-based image generation.
        - Pay attention to details such as background color, text color, font size, padding, margins, borders, and other visual elements.
        - **Do not omit any module's code**. Every module, regardless of its size or complexity, must be included in the final HTML code. Ensure that each module's functionality and layout are represented fully in the assembled page.
                         
        # Libraries #
        - You may use Google Fonts to match the fonts in the screenshot.
        - Use the Font Awesome icon library: <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"></link>
        """)

class AgentRepairA11y(MLLMAgent):
    def __init__(self):
        super().__init__("""
You are an accessibility remediation agent for HTML.

Input format:
You will receive two text inputs in the user message:
1) Full original HTML code string.
2) One accessibility report JSON string.

Your job:
- Analyze issues from the provided report only.
- Repair the HTML to reduce or remove those violations.
- Preserve visual layout and content as much as possible.
- Do not remove meaningful content or major structure just to pass checks.
- Prefer semantic and accessible fixes: landmarks, labels, alt text, roles, aria attributes, heading order, contrast-friendly style tweaks, form associations, button/link names, etc.

Important constraints:
- Return full, runnable HTML.
- Keep external libraries already present.
- Keep existing text/content unless change is required for accessibility.
- Avoid overusing ARIA when native semantic HTML solves the same issue.

Return format (strict):
```json
{
  "analysis": {
    "violation_count": <int>,
    "key_issues": ["..."],
    "fix_strategy": ["..."]
  },
  "changes": [
    {
      "issue_id": "axe_rule_id_or_custom",
      "what_changed": "short explanation"
    }
  ],
  "repaired_html": "<full html string>"
}
```
""")

    def infer(self, html_text, report_json_text, parse=True, temperature=0, seed=SEED, n=1):
        user_text = [
            "Input 1: Original HTML code:\n" + html_text + 
            "\n\nInput 2: Accessibility report JSON\n" +
            report_json_text,
        ]
        # user_text= ["Reply with exactly: OK"]
        return super().infer(user_text, parse=parse, temperature=temperature, seed=seed, n=n)

    def parser(self, text):
        data = _parse_json_object(text)
        if not isinstance(data, dict):
            data = {}

        if "repaired_html" not in data:
            html_block = _extract_first_html_block(text)
            if html_block:
                data["repaired_html"] = html_block

        return data


class AgentMergeA11yReports(MLLMAgent):
    def __init__(self):
        super().__init__("""
You are an accessibility report consolidation agent.

Input format:
You will receive one JSON object containing:
1) `axe_report`: JSON object (or null)
2) `lighthouse_report`: JSON object (or null)
3) `pa11y_report`: JSON array/object (or null)

Your job:
- Merge findings from all available tools into one unified JSON report.
- Deduplicate overlapping issues by semantic meaning and target selector when possible.
- Keep track of source tools for each merged issue.
- Prioritize concrete accessibility problems over informational checks.

Output format (strict JSON):
```json
{
  "summary": {
    "tools_received": {
      "axe": <bool>,
      "lighthouse": <bool>,
      "pa11y": <bool>
    },
    "total_merged_issues": <int>,
    "by_severity": {
      "critical": <int>,
      "serious": <int>,
      "moderate": <int>,
      "minor": <int>,
      "unknown": <int>
    }
  },
  "merged_issues": [
    {
      "issue_id": "<stable id>",
      "title": "<short issue title>",
      "description": "<clear explanation>",
      "severity": "<critical|serious|moderate|minor|unknown>",
      "selectors": ["..."],
      "wcag": ["..."],
      "source_tools": ["axe", "lighthouse", "pa11y"],
      "source_rule_ids": ["..."],
      "recommended_fix": "<actionable fix>"
    }
  ]
}
```
""")

    def parser(self, text):
        data = _parse_json_object(text)
        return data if isinstance(data, dict) else {}



if __name__ == "__main__":
    import json
    from PIL import Image
    from openai_client import gpt
    BACKBONE = gpt
    agent = AgentAssemble()
    with open("output/modules_repaired.json", "r") as f:
        test_input = json.load(f)
    image = Image.open("output/input.png")
    output = agent.infer([image, test_input], parse=True)
    print(output)
