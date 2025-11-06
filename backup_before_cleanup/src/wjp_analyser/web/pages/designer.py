import base64
import os
import sys
import numpy as np
import cv2
from pathlib import Path

import streamlit as st

# Add project root to path for agent imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # Import agents from the correct path
    import os
    # Change to project root directory
    original_dir = os.getcwd()
    project_root_str = str(project_root)
    os.chdir(project_root_str)
    
    # Add wjp_agents to path
    wjp_agents_path = os.path.join(project_root_str, "wjp_agents")
    if wjp_agents_path not in sys.path:
        sys.path.insert(0, wjp_agents_path)
    
    # Import agents
    from wjp_agents.designer_agent import DesignerAgent
    from wjp_agents.image_to_dxf_agent import ImageToDXFAgent
    from wjp_agents.learning_agent import LearningAgent
    
    # Restore original directory
    os.chdir(original_dir)
    AGENTS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Agent system not available: {e}")
    AGENTS_AVAILABLE = False


st.title("Designer (Prompt -> Image)")

# Helper for guided flow (defined before usage to avoid NameError)
def build_prompt_ui(form) -> str:
    WJ_RULES = (
        "top-view, centered, flat orthographic, no perspective; "
        "waterjet-safe geometry (>=3 mm spacing, >=2 mm inner radius, min 5 mm segment); "
        "clear contrasting inlays; simple, bold shapes; avoid hairline filigree."
    )
    size_text = f"dimensions {form['dimensions']}" if form.get("dimensions") else ""
    sym = "perfectly symmetrical" if form.get("symmetry_required", True) else "asymmetrical allowed"
    safety = "waterjet-safe" if form.get("waterjet_safe", True) else "decorative-only"
    base = (
        f"High-resolution {('top-view ' if form.get('top_view', True) else '')}{str(form['category']).lower()} "
        f"design in {form['material']}. {size_text}. "
        f"Style: {form['style']}. Complexity {form['complexity']}/5. {sym}. {safety}. "
        f"{WJ_RULES} "
    )
    if form.get("notes"):
        base += f"Notes: {form['notes']}. "
    base += (
        "white studio background; crisp edges; no shadows; "
        "export as 1:1 square frame; centered; "
        "avoid text, logos, and photoreal grain - use clean vector-like shapes."
    )
    return base

# Conversational guided flow (assistant asks, user answers)
with st.container():
    st.markdown("---")
    st.subheader("Interactive Assistant")
    start_guided = st.checkbox("Talk to the Designer Assistant", value=False, help="Step-by-step Q&A to build your prompt")
    if start_guided:
        # Initialize wizard state
        if "designer_wizard" not in st.session_state:
            st.session_state.designer_wizard = {
                "step": 0,
                "answers": {
                    "category": None,
                    "material": None,
                    "dimensions": None,
                    "style": None,
                    "complexity": 3,
                    "top_view": True,
                    "symmetry_required": True,
                    "waterjet_safe": True,
                    "notes": "",
                    "variations": 1,
                    "negative_hints": "hairline filigree, text, logos, shadows",
                }
            }

        wiz = st.session_state.designer_wizard

        steps = [
            "category", "material", "dimensions", "style", "constraints",
            "notes", "options", "review"
        ]

        def next_step():
            wiz["step"] = min(wiz["step"] + 1, len(steps) - 1)

        def prev_step():
            wiz["step"] = max(wiz["step"] - 1, 0)

        st.caption(f"Step {wiz['step']+1} of {len(steps)}")

        with st.chat_message("assistant"):
            if steps[wiz["step"]] == "category":
                st.markdown("What are you designing today?")
                wiz["answers"]["category"] = st.radio(
                    "Choose a category",
                    ["Inlay Tile","Jali Panel","Drain Cover","Medallion","Signage","Industrial"],
                    index=["Inlay Tile","Jali Panel","Drain Cover","Medallion","Signage","Industrial"].index(wiz["answers"].get("category") or "Inlay Tile")
                )
            elif steps[wiz["step"]] == "material":
                st.markdown("Select the base material")
                wiz["answers"]["material"] = st.radio(
                    "Material",
                    ["Tan Brown Granite","White Marble","Black Granite","Jaisalmer Yellow"],
                    index=["Tan Brown Granite","White Marble","Black Granite","Jaisalmer Yellow"].index(wiz["answers"].get("material") or "Tan Brown Granite")
                )
            elif steps[wiz["step"]] == "dimensions":
                st.markdown("Provide target size (e.g., 2x2 ft or 600x600 mm)")
                wiz["answers"]["dimensions"] = st.text_input("Dimensions", value=wiz["answers"].get("dimensions") or "2x2 ft")
            elif steps[wiz["step"]] == "style":
                st.markdown("Pick a style direction")
                preset = st.selectbox("Style presets", [
                    "geometric floral lattice, bold outlines",
                    "minimal concentric rings, high contrast",
                    "traditional Indian medallion, simple inlays",
                    "modern linear grooves, clean arcs"
                ])
                wiz["answers"]["style"] = st.text_input("Style (editable)", value=wiz["answers"].get("style") or preset)
            elif steps[wiz["step"]] == "constraints":
                st.markdown("Set cutting constraints")
                c1, c2, c3 = st.columns(3)
                with c1:
                    wiz["answers"]["complexity"] = st.slider("Complexity", 1, 5, int(wiz["answers"].get("complexity") or 3))
                with c2:
                    wiz["answers"]["top_view"] = st.checkbox("Top-view", bool(wiz["answers"].get("top_view", True)))
                with c3:
                    wiz["answers"]["symmetry_required"] = st.checkbox("Symmetry required", bool(wiz["answers"].get("symmetry_required", True)))
                wiz["answers"]["waterjet_safe"] = st.checkbox("Waterjet-safe geometry", bool(wiz["answers"].get("waterjet_safe", True)))
            elif steps[wiz["step"]] == "notes":
                st.markdown("Any extra notes or references?")
                wiz["answers"]["notes"] = st.text_area("Notes", value=wiz["answers"].get("notes") or "")
            elif steps[wiz["step"]] == "options":
                st.markdown("Generation options")
                oc1, oc2 = st.columns(2)
                with oc1:
                    wiz["answers"]["variations"] = st.number_input("Variations", 1, 4, int(wiz["answers"].get("variations") or 1))
                with oc2:
                    wiz["answers"]["negative_hints"] = st.text_input("Avoid", value=wiz["answers"].get("negative_hints") or "hairline filigree, text, logos, shadows")
            elif steps[wiz["step"]] == "review":
                st.markdown("Review your selections")
                st.json(wiz["answers"])
                # Build prompt preview
                preview_prompt = build_prompt_ui({
                    "category": wiz["answers"]["category"],
                    "material": wiz["answers"]["material"],
                    "dimensions": wiz["answers"]["dimensions"],
                    "style": wiz["answers"]["style"],
                    "complexity": wiz["answers"]["complexity"],
                    "notes": wiz["answers"]["notes"],
                    "top_view": wiz["answers"]["top_view"],
                    "symmetry_required": wiz["answers"]["symmetry_required"],
                    "waterjet_safe": wiz["answers"]["waterjet_safe"],
                })
                st.code(preview_prompt)
                gen = st.button("Generate Now", type="primary")
                if gen and AGENTS_AVAILABLE:
                    try:
                        designer = DesignerAgent()
                        opts = {
                            "variations": int(wiz["answers"]["variations"]),
                            "negative_hints": wiz["answers"]["negative_hints"],
                        }
                        res = designer.run(preview_prompt, options=opts)
                        img_paths = res.get("image_paths") or []
                        if img_paths:
                            st.success("Images generated")
                            st.session_state["designer_image_path"] = img_paths[0]
                            # Show grid
                            cols = st.columns(min(3, len(img_paths)))
                            for i, p in enumerate(img_paths):
                                with cols[i % len(cols)]:
                                    if os.path.exists(p):
                                        st.image(p, caption=os.path.basename(p), use_column_width=True)
                    except Exception as e:
                        st.error(f"Generation failed: {e}")

        nav1, nav2 = st.columns([1,1])
        with nav1:
            if st.button("◀ Back", disabled=wiz["step"] == 0):
                prev_step()
        with nav2:
            if st.button("Next ▶", disabled=wiz["step"] >= len(steps)-1):
                next_step()

        st.markdown("---")

# Live Preview: show last generated image or latest from output directory
try:
    preview_path = None
    # Prefer session image if present
    if hasattr(st, "session_state") and st.session_state and st.session_state.get("designer_image_path"):
        candidate = st.session_state.get("designer_image_path")
        if candidate and os.path.exists(candidate):
            preview_path = candidate
    # Else pick latest PNG from output/designer
    if preview_path is None:
        out_dir = Path("output/designer")
        if out_dir.exists():
            pngs = sorted(out_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pngs:
                preview_path = str(pngs[0])
    if preview_path and os.path.exists(preview_path):
        st.markdown("---")
        st.subheader("Live Preview")
        st.image(preview_path, caption=os.path.basename(preview_path), use_column_width=True)
        st.caption("Latest generated design preview")

        # Transform & Orientation controls
        with st.expander("Transform & Orientation", expanded=False):
            st.caption("Adjust orientation before conversion. Saves a new image copy when applied.")

            def _load_bgr(path: str):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                return img

            def _rotate_image(bgr, angle_deg: float):
                h, w = bgr.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
                rotated = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                return rotated

            def _tilt_image(bgr, tilt_x: float, tilt_y: float):
                # Perspective warp approximating angle-of-view tilt
                h, w = bgr.shape[:2]
                dx = int(w * tilt_x * 0.25)
                dy = int(h * tilt_y * 0.25)
                src = np.float32([[0,0],[w,0],[w,h],[0,h]])
                dst = np.float32([[0+dx,0+dy],[w-dx,0+dy],[w-dx,h-dy],[0+dx,h-dy]])
                P = cv2.getPerspectiveTransform(src, dst)
                warped = cv2.warpPerspective(bgr, P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                return warped

            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                angle = st.slider("Rotation (°)", -180, 180, 0, 1)
            with c2:
                tilt_x = st.slider("Tilt X", -0.5, 0.5, 0.0, 0.01)
            with c3:
                tilt_y = st.slider("Tilt Y", -0.5, 0.5, 0.0, 0.01)

            show_preview = st.button("Preview Transform")
            save_transformed = st.button("Save Transformed Copy")

            if show_preview or save_transformed:
                try:
                    bgr = _load_bgr(preview_path)
                    if bgr is None:
                        st.error("Failed to load image for transform")
                    else:
                        out = _rotate_image(bgr, angle) if angle else bgr
                        if abs(tilt_x) > 1e-6 or abs(tilt_y) > 1e-6:
                            out = _tilt_image(out, float(tilt_x), float(tilt_y))
                        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                        st.image(rgb, caption="Transformed Preview", use_column_width=True)

                        if save_transformed:
                            out_dir = Path("output/designer")
                            out_dir.mkdir(parents=True, exist_ok=True)
                            suffix = f"rot{int(angle)}_tx{int(tilt_x*100)}_ty{int(tilt_y*100)}"
                            save_path = out_dir / f"{Path(preview_path).stem}_{suffix}.png"
                            cv2.imwrite(str(save_path), out)
                            st.success(f"Saved: {save_path}")
                            if hasattr(st, "session_state"):
                                st.session_state["designer_image_path"] = str(save_path)
                except Exception as _e:
                    st.error(f"Transform failed: {_e}")
        # Recent gallery (up to 6 thumbnails)
        out_dir = Path("output/designer")
        if out_dir.exists():
            pngs = sorted(out_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:6]
            if pngs:
                st.markdown("#### Recent Outputs")
                cols = st.columns(min(3, len(pngs)))
                for i, p in enumerate(pngs):
                    with cols[i % len(cols)]:
                        st.image(str(p), caption=p.name, use_column_width=True)
except Exception:
    pass


def build_prompt(form) -> str:
    WJ_RULES = (
        "top-view, centered, flat orthographic, no perspective; "
        "waterjet-safe geometry (>=3 mm spacing, >=2 mm inner radius, min 5 mm segment); "
        "clear contrasting inlays; simple, bold shapes; avoid hairline filigree."
    )
    size_text = f"dimensions {form['dimensions']}" if form.get("dimensions") else ""
    sym = "perfectly symmetrical" if form.get("symmetry_required", True) else "asymmetrical allowed"
    safety = "waterjet-safe" if form.get("waterjet_safe", True) else "decorative-only"
    base = (
        f"High-resolution {('top-view ' if form.get('top_view', True) else '')}{str(form['category']).lower()} "
        f"design in {form['material']}. {size_text}. "
        f"Style: {form['style']}. Complexity {form['complexity']}/5. {sym}. {safety}. "
        f"{WJ_RULES} "
    )
    if form.get("notes"):
        base += f"Notes: {form['notes']}. "
    base += (
        "white studio background; crisp edges; no shadows; "
        "export as 1:1 square frame; centered; "
        "avoid text, logos, and photoreal grain - use clean vector-like shapes."
    )
    return base


with st.sidebar.form("designer"):
    with st.expander("Design Details", expanded=True):
        category = st.selectbox("Category", ["Inlay Tile","Jali Panel","Drain Cover","Medallion","Signage","Industrial"])
        material = st.selectbox("Material", ["Tan Brown Granite","White Marble","Black Granite","Jaisalmer Yellow"])
        dimensions = st.text_input("Dimensions", help="e.g., 2x2 ft or 600x600 mm")
        style = st.text_input("Pattern Style", value="geometric floral lattice, bold outlines")
    with st.expander("Constraints", expanded=False):
        complexity = st.slider("Design Complexity", 1, 5, 3)
        colA, colB, colC = st.columns(3)
        with colA:
            top_view = st.checkbox("Top-view", True)
        with colB:
            symmetry_required = st.checkbox("Symmetry required", True)
        with colC:
            waterjet_safe = st.checkbox("Waterjet-safe", True)
    with st.expander("Notes", expanded=False):
        notes = st.text_area("Notes (optional)")
    submitted = st.form_submit_button("Build Prompt")

if submitted:
    form = {
        "category": category, "material": material, "dimensions": dimensions,
        "style": style, "complexity": complexity, "notes": notes,
        "top_view": top_view, "symmetry_required": symmetry_required, "waterjet_safe": waterjet_safe,
    }
    prompt = build_prompt(form)
    st.subheader("Prompt Preview")
    st.code(prompt, language="text")
    try:
        st.download_button("Download Prompt (.txt)", data=prompt.encode('utf-8'), file_name="design_prompt.txt")
    except Exception:
        pass

    # Enhanced image generation with agent system
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Image (Enhanced)", type="primary"):
            if AGENTS_AVAILABLE:
                try:
                    # Use DesignerAgent to create a real image
                    import os
                    original_dir = os.getcwd()
                    project_root_str = str(project_root)
                    os.chdir(project_root_str)
                    
                    # Add wjp_agents to path
                    wjp_agents_path = os.path.join(project_root_str, "wjp_agents")
                    if wjp_agents_path not in sys.path:
                        sys.path.insert(0, wjp_agents_path)
                    
                    # Debug info
                    st.info(f"🔧 Working directory: {os.getcwd()}")
                    st.info(f"🔧 Project root: {project_root_str}")
                    
                    designer = DesignerAgent()
                    # Advanced options
                    with st.expander("Advanced Generation Options", expanded=False):
                        v_cols = st.columns([1,1])
                        with v_cols[0]:
                            variations = st.number_input("Variations", min_value=1, max_value=4, value=1, step=1)
                        with v_cols[1]:
                            negative_hints = st.text_input("Negative hints (Avoid)", value="hairline filigree, text, logos, shadows")
                    user_input = f"{category} in {material}, {style}, complexity {complexity}/5"
                    if notes:
                        user_input += f", {notes}"
                    
                    st.info(f"🎨 Generating design for: {user_input}")
                    
                    result = designer.run(user_input, options={
                        "variations": variations,
                        "negative_hints": negative_hints,
                    })
                    image_paths = result.get("image_paths") or []
                    image_path = image_paths[0] if image_paths else None
                    
                    st.info(f"📁 Generated image path: {image_path}")
                    
                    # Restore original directory
                    os.chdir(original_dir)
                    
                    # Display the generated image
                    st.subheader("Generated Image(s)")
                    if image_path and os.path.exists(image_path):
                        # Grid for variations
                        if len(image_paths) > 1:
                            cols = st.columns(min(3, len(image_paths)))
                            for i, p in enumerate(image_paths):
                                with cols[i % len(cols)]:
                                    if os.path.exists(p):
                                        st.image(p, caption=os.path.basename(p), use_column_width=True)
                        else:
                            st.image(image_path, caption="AI Generated Design", use_column_width=True)
                        st.success("✅ Image generation complete")
                    else:
                        st.error(f"❌ Image file not found at: {image_path}")
                        st.info("The DesignerAgent generated the image, but the file path may be incorrect.")
                    
                    # Store image path for further processing
                    if image_path:
                        st.session_state["designer_image_path"] = image_path
                    st.session_state["designer_prompt"] = prompt
                    
                    # Download button
                    if image_path and os.path.exists(image_path):
                        with open(image_path, "rb") as f:
                            image_data = f.read()
                        st.download_button("Download First Image", data=image_data, file_name="designer_image.png")
                    
                    # Additional debugging info
                    st.info(f"📁 Output directory: {designer.output_dir}")
                    st.info(f"📄 Generated file: {os.path.basename(image_path)}")
                    st.success("✅ Image generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating image: {e}")
            else:
                st.error("Agent system not available. Please check the installation.")
    
    with col2:
        if st.button("Generate Image (Stub)"):
            out_dir = Path("data/outputs/images")
            out_dir.mkdir(parents=True, exist_ok=True)
            p = out_dir / "designer_stub.png"
            if not p.exists():
                p.write_bytes(base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
                ))
            st.image(str(p), caption="Generated (stub)")
            try:
                st.download_button("Download Image", data=p.read_bytes(), file_name="designer_image.png")
            except Exception:
                pass
            st.session_state["designer_image_path"] = str(p)

    # Pipeline integration
    if st.session_state.get("designer_image_path"):
        st.markdown("---")
        st.subheader("Pipeline Integration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Send to Image → DXF", type="secondary"):
                st.session_state["img2dxf_image_path"] = st.session_state["designer_image_path"]
                st.success("✅ Image queued for Image→DXF conversion!")
                st.info("Navigate to the 'Image to DXF' page to continue.")
        
        with col2:
            if st.button("Run Agent Pipeline", type="secondary") and AGENTS_AVAILABLE:
                try:
                    with st.spinner("Running full agent pipeline..."):
                        # Change to project root for agent execution
                        import os
                        original_dir = os.getcwd()
                        project_root_str = str(project_root)
                        os.chdir(project_root_str)
                        
                        # Add wjp_agents to path
                        wjp_agents_path = os.path.join(project_root_str, "wjp_agents")
                        if wjp_agents_path not in sys.path:
                            sys.path.insert(0, wjp_agents_path)
                        
                        try:
                            # Initialize agents
                            designer = DesignerAgent()
                            converter = ImageToDXFAgent()
                            learner = LearningAgent()
                            
                            # Run the pipeline
                            image_path = st.session_state["designer_image_path"]
                            
                            # Convert to DXF with optimization
                            st.write("🔄 Converting image to DXF...")
                            dxf_result = converter.run(image_path)
                            
                            # Run parameter optimization
                            st.write("🧠 Optimizing parameters...")
                            optimization_result = learner.run(image_path, max_iterations=10)
                            
                            # Display results
                            st.success("✅ Agent pipeline completed!")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Objects Detected", dxf_result.get("objects_detected", 0))
                                st.metric("DXF File", "Generated")
                            
                            with col_b:
                                st.metric("Best Score", f"{optimization_result.get('best_score', 0):.2f}")
                                st.metric("Optimization", "Complete")
                            
                            # Show best parameters
                            if optimization_result.get("best_params"):
                                st.subheader("Optimized Parameters")
                                params = optimization_result["best_params"]
                                st.json({
                                    "min_area": params.min_area,
                                    "max_area": params.max_area,
                                    "min_circularity": params.min_circularity,
                                    "min_solidity": params.min_solidity,
                                    "merge_distance": params.merge_distance
                                })
                        
                        finally:
                            # Restore original directory
                            os.chdir(original_dir)
                        
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
        
        with col3:
            if st.button("Preview Pipeline", type="secondary") and AGENTS_AVAILABLE:
                try:
                    st.subheader("Pipeline Preview")
                    
                    # Show what the pipeline would do
                    st.info("""
                    **Full Agent Pipeline Preview:**
                    
                    1. **DesignerAgent** → Generate design image
                    2. **ImageToDXFAgent** → Convert to DXF with object detection
                    3. **LearningAgent** → Optimize parameters (10 iterations)
                    4. **AnalyzeDXFAgent** → Analyze quality metrics
                    5. **ReportAgent** → Generate comprehensive report
                    
                    **Expected Output:**
                    - Optimized DXF file
                    - Quality analysis report
                    - Parameter optimization results
                    - Preview images
                    """)
                    
                except Exception as e:
                    st.error(f"Preview error: {e}")

