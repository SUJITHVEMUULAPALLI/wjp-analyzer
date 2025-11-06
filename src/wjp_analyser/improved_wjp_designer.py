"""
Improved WJP Designer UI
------------------------

This Streamlit application collects user intent to generate DXF‑ready prompts and
preview clean images for waterjet and CNC designs. The interface has been
re‑structured based on user‑friendly design principles, including:

* **Clear hierarchy and instructions:** Each step is clearly labeled with a
  heading and short description so users know what to do and why. Long
  paragraphs are avoided to keep text scannable【1499181242979†L241-L246】.
* **Logical grouping of inputs:** All inputs for describing the design intent are
  contained within a form. This prevents accidental reloads and groups
  related fields together, making it easier to complete the task【66364925532471†L359-L363】.
* **Plain language and examples:** Help text accompanies each field to
  clarify what type of input is expected and provide examples in the user’s
  language【185390840779846†L316-L318】.
* **Accessible labels:** Each input includes a descriptive label. While
  Streamlit automatically associates labels, explicit descriptions ensure
  assistive technologies can interpret the form【515953048535652†L262-L269】.
* **Step‑by‑step actions:** The app divides the process into steps
  (Describe → Generate Prompts → Preview Image → Download/Convert) with
  clear call‑to‑action buttons【1499181242979†L300-L308】.
* **Performance hints:** Prompts are generated locally and images are
  generated only upon user request, avoiding unnecessary API calls and
  improving perceived speed【1499181242979†L414-L421】.

"""

import os
import base64
import streamlit as st

# OpenAI client is optional at import time; handle missing dependency gracefully
try:
    from openai import OpenAI  # type: ignore
except Exception:  # ImportError or version issues
    OpenAI = None  # type: ignore


def generate_prompt_variants(
    design_type: str,
    style: str,
    shape: str,
    material: str,
    size: str,
    detail: str,
    goal: str,
) -> dict:
    """Create three DXF‑ready prompt variants from the user intent.

    Parameters
    ----------
    design_type : str
        The type of design (e.g. Tile, Medallion).
    style : str
        Style keywords such as geometric, floral or modern.
    shape : str
        Overall shape or layout (e.g. circular, rectangular).
    material : str
        Material for manufacturing (e.g. Tan Brown Granite, Steel).
    size : str
        Frame size in millimetres (defaults to 1000 x 1000).
    detail : str
        Level of detail (Minimal, Moderate, Highly Detailed).
    goal : str
        The intended use (e.g. Waterjet Inlay).

    Returns
    -------
    dict
        A dictionary of prompt variants keyed by layer name.
    """
    # Base description uses plain language and echoes the user intent
    base_desc = (
        f"Create a {detail.lower()} {style.lower()} {design_type.lower()} "
        f"design in a {shape.lower()} layout. "
        f"The design is intended for a {material} {goal.lower()} and should fit within "
        f"a {size} frame. "
    )

    # DXF safety guidelines ensure vector conversion works cleanly
    dxf_guidelines = (
        "Render a clean technical line drawing suitable for waterjet or CNC cutting. "
        "Use only black lines on a pure white background. Avoid gradients, colors, shadows, "
        "or textures. Ensure all contours are continuous and closed. Minimum spacing is 3 mm "
        "and the smallest inner curve radius is 2 mm. Center and scale the drawing within "
        "a 1000 mm × 1000 mm frame."
    )

    # Styles for each variant; these are appended to differentiate outline and fill
    variant_styles = {
        "OUTLINE_ONLY": "Use thin black outlines only; no fills.",
        "FILLED_BLACK": "Render solid black filled regions representing cut areas; no outlines.",
        "HYBRID": "Combine filled black areas with thin outlines for clarity.",
    }

    # Assemble final prompts with meta info
    prompts = {}
    for variant, variant_style in variant_styles.items():
        meta = (
            f"[Design Type: {design_type.upper()}] "
            f"[Material: {material}] "
            f"[Layer: {variant}] [Purpose: DXF conversion]"
        )
        prompts[variant] = (
            f"{base_desc}{dxf_guidelines} {variant_style} {meta}"
        )
    return prompts


def main() -> None:
    """Runs the Streamlit app."""
    # Guard against double page-config when embedded in a parent Streamlit app
    try:
        st.set_page_config(page_title="WJP Designer – DXF Prompt & Image", layout="centered")
    except Exception:
        pass
    st.title("🧩 WJP Designer – Generate DXF‑Ready Designs")

    # Introduction with brief explanation. Use bullet points for scannability.
    st.markdown(
        """
        Design custom patterns for waterjet and CNC cutting. Follow the steps below:
        
        1. **Describe your design** – Tell us what you want to create.
        2. **Generate prompts** – We'll convert your intent into AI prompts for DALL·E.
        3. **Preview clean images** – Choose a variant and see a high‑contrast preview.
        4. **Download & convert** – Save the image and feed it into your DXF converter.
        """
    )

    # Step 1: User intent form
    st.header("Step 1: Describe your design")
    st.caption(
        "Provide details about the design you want. Fields marked with * are required."
    )

    with st.form("intent_form"):
        col1, col2 = st.columns(2)
        with col1:
            design_type = st.selectbox(
                "Design type*",
                ["Tile", "Medallion", "Jali", "Grill", "Border", "Custom"],
                help="Select the general category of your design."
            )
            style = st.text_input(
                "Style keywords*",
                value="Geometric",
                help="Describe the style – e.g., geometric, floral, traditional."
            )
            shape = st.text_input(
                "Shape / layout*",
                value="Circular",
                help="Specify the overall shape of the design – e.g., circular, rectangular."
            )
        with col2:
            material = st.selectbox(
                "Material*",
                [
                    "Tan Brown Granite",
                    "White Marble",
                    "Black Granite",
                    "Steel",
                    "Custom",
                ],
                help="Choose the material you will use."
            )
            size = st.text_input(
                "Frame size (mm)*",
                value="1000 x 1000",
                help="Enter the frame dimensions, e.g., 1000 x 1000."
            )
            detail = st.selectbox(
                "Detail level*",
                ["Minimal", "Moderate", "Highly Detailed"],
                help="Select how intricate the pattern should be."
            )
        goal = st.selectbox(
            "Output goal*",
            ["Waterjet Inlay", "Metal Grill", "Drain Cover", "Decorative Tile", "Custom"],
            help="What will this design be used for?"
        )

        submit = st.form_submit_button("Generate DXF‑ready prompts")

    # When the form is submitted, generate prompts
    if submit:
        # Basic validation: ensure required fields are filled
        if not (design_type and style and shape and material and size and detail and goal):
            st.error("Please complete all required fields.")
        else:
            prompts = generate_prompt_variants(
                design_type=design_type,
                style=style,
                shape=shape,
                material=material,
                size=size,
                detail=detail,
                goal=goal,
            )
            st.success("Prompts generated successfully. Proceed to Step 2.")
            st.session_state["prompts"] = prompts

    # Step 2: Show generated prompts and allow selection
    if "prompts" in st.session_state:
        st.header("Step 2: Review your prompts")
        st.caption(
            "Select a variant to generate a preview image. The variant defines the line and fill style "
            "for better conversion results."
        )
        # Present prompts in an expander to reduce clutter
        with st.expander("Show all generated prompts", expanded=False):
            for layer, prompt in st.session_state["prompts"].items():
                st.subheader(f"Variant: {layer}")
                st.text_area(
                    label=f"Prompt for {layer}",
                    value=prompt,
                    height=160,
                    help="This is the text sent to DALL·E to produce your design."
                )

        selected_variant = st.radio(
            "Choose a variant for image generation",
            list(st.session_state["prompts"].keys()),
            help="Outline only: black lines; Filled black: black regions; Hybrid: mix of lines and fill."
        )

        # Step 3: Generate image
        st.header("Step 3: Generate your clean image")
        if st.button("Generate clean image"):
            with st.spinner("Generating clean image…"):
                # Validate OpenAI availability and API key
                if OpenAI is None:
                    st.error("OpenAI client not available. Please install the 'openai' package as per requirements.txt.")
                    return

                api_key = os.environ.get("OPENAI_API_KEY")
                # Streamlit secrets support (optional)
                if not api_key and hasattr(st, "secrets"):
                    try:
                        api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
                    except Exception:
                        api_key = None

                if not api_key:
                    st.error("OpenAI API key not found. Set environment variable OPENAI_API_KEY or enter it below.")
                    entered = st.text_input("Enter OpenAI API Key", type="password", help="Your key is kept only in this session.")
                    if st.button("Save API Key for this session") and entered:
                        os.environ["OPENAI_API_KEY"] = entered
                        st.success("API key saved for this session. Click 'Generate clean image' again.")
                    return

                client = OpenAI()
                dalle_prompt = st.session_state["prompts"][selected_variant]
                try:
                    image_response = client.images.generate(
                        model="gpt-image-1",
                        prompt=dalle_prompt,
                        size="1024x1024",
                        n=1,
                        response_format="b64_json",
                    )
                    datum = image_response.data[0]
                    # Support both SDK object and dict responses
                    if isinstance(datum, dict):
                        image_url = datum.get("url")
                        b64_data = datum.get("b64_json")
                    else:
                        image_url = getattr(datum, "url", None)
                        b64_data = getattr(datum, "b64_json", None)

                    if b64_data:
                        try:
                            img_bytes = base64.b64decode(b64_data)
                        except Exception as _decode_err:
                            raise ValueError(f"Failed to decode image bytes: {_decode_err}")
                        if not img_bytes:
                            raise ValueError("Empty image bytes returned by API")
                        st.image(
                            img_bytes,
                            caption=f"Preview: {selected_variant}",
                            width="stretch",
                        )
                        st.session_state["generated_image_bytes"] = img_bytes
                        st.session_state.pop("generated_image_url", None)
                    elif image_url:
                        st.image(
                            image_url,
                            caption=f"Preview: {selected_variant}",
                            width="stretch",
                        )
                        st.session_state["generated_image_url"] = image_url
                        st.session_state.pop("generated_image_bytes", None)
                    else:
                        raise ValueError("No image returned by API (no url or b64 data)")

                    st.success("Image generated successfully. Proceed to Step 4.")
                except Exception as e:
                    st.error(f"An error occurred while generating the image: {e}")
                    st.caption("Tip: If this persists, try again or adjust your prompt for simpler geometry.")

    # Step 4: Download or convert
    if "generated_image_url" in st.session_state:
        st.header("Step 4: Download & Convert")
        st.write(
            "Your image is ready for vector tracing and DXF conversion. "
            "Download the image and upload it to your preferred converter (e.g., Inkscape or your own pipeline)."
        )

        if st.session_state.get("generated_image_url"):
            st.markdown(
                f"[Download clean image]({st.session_state['generated_image_url']})",
                unsafe_allow_html=True,
            )
        elif st.session_state.get("generated_image_bytes"):
            st.download_button(
                label="Download clean image",
                data=st.session_state["generated_image_bytes"],
                file_name="designer_image.png",
                mime="image/png",
            )

        st.write(
            "After downloading, you can proceed to convert it into a DXF file using your chosen tool. "
            "If you are using our integrated converter, upload this image there."
        )


if __name__ == "__main__":
    main()