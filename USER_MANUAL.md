# Waterjet DXF Analyzer - User Manual

## 🚀 Getting Started

### Quick Start
1. **Access the Web Interface**: Open http://127.0.0.1:5000 in your browser
2. **Choose Your Workflow**: Select from the main menu options
3. **Upload Files**: Drag and drop or browse for your DXF/image files
4. **Configure Settings**: Adjust parameters as needed
5. **Run Analysis**: Click the analyze button and wait for results
6. **Download Results**: Get your reports, files, and quotes

### System Requirements
- **Web Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **Internet Connection**: Required for AI analysis (optional for basic features)
- **File Formats**: DXF files for analysis, PNG/JPG for image conversion
- **Potrace**: Required for advanced image-to-DXF conversion (install separately)
  - **Windows**: Download from http://potrace.sourceforge.net/ or install via Chocolatey: `choco install potrace`
  - **Linux**: `sudo apt-get install potrace` (Ubuntu/Debian) or `sudo yum install potrace` (RHEL/CentOS)
  - **macOS**: `brew install potrace`

## 📋 Main Workflows

### 1. DXF Analysis
**Purpose**: Analyze DXF files for waterjet cutting feasibility and cost estimation

**Steps**:
1. Click **"1. DXF Analysis"** from the main menu
2. Upload your DXF file using the file browser
3. Configure analysis parameters:
   - **Material**: Select material type (steel, aluminum, etc.)
   - **Thickness**: Set material thickness in mm
   - **Kerf**: Enter kerf width (typically 0.1-0.3mm)
   - **Cutting Speed**: Set cutting speed in mm/min
   - **Cost Rate**: Enter cost per meter
4. Click **"Analyze DXF"**
5. Review the results:
   - **Cutting Length**: Total cutting distance
   - **Pierce Points**: Number of entry points
   - **Estimated Time**: Cutting time estimate
   - **Cost Estimate**: Material and cutting costs
   - **Quality Issues**: Any manufacturing concerns
6. Download reports, preview images, or generate quotes

**Use Cases**:
- Pre-production feasibility analysis
- Cost estimation for quotes
- Quality validation before cutting
- Process optimization

### 2. AI Analysis
**Purpose**: Get AI-powered manufacturing insights and recommendations

**Steps**:
1. Click **"5. AI Analysis"** from the main menu
2. Upload your DXF file
3. Select AI model:
   - **Auto**: Automatically selects the best available model
   - **Ollama Models**: Local AI models (waterjet:latest, gpt-oss:20b)
   - **OpenAI**: Cloud-based GPT-4 (requires API key)
4. Set timeout (2-10 minutes)
5. Click **"Run AI Analysis"**
6. Review AI insights:
   - **Feasibility Score**: 0-100 manufacturing score
   - **Complexity Level**: Simple/Moderate/Complex
   - **Time Estimate**: AI-calculated cutting time
   - **Material Recommendations**: Suggested materials
   - **Toolpath Suggestions**: Cutting strategies
   - **Potential Issues**: Manufacturing concerns
   - **Optimization Tips**: Efficiency improvements
   - **Cost Considerations**: Budget insights

**Use Cases**:
- Manufacturing feasibility assessment
- Process optimization recommendations
- Quality improvement suggestions
- Cost optimization strategies

### 3. Nesting
**Purpose**: Arrange multiple DXF files on material sheets for optimal material usage

**Steps**:
1. Click **"4. Nesting"** from the main menu
2. Upload multiple DXF files (use Ctrl+click to select multiple)
3. Configure sheet parameters:
   - **Sheet Width**: Material sheet width in mm
   - **Sheet Height**: Material sheet height in mm
   - **Spacing**: Distance between parts in mm
4. Click **"Inspect Files"**
5. Review the file inspection:
   - **Part Dimensions**: Size of each uploaded part
   - **Quantity**: Set quantity for each part
   - **Fit Check**: Verify parts fit on the sheet
6. Click **"Confirm Nesting"**
7. Review nesting results:
   - **Sheet Utilization**: Percentage of material used
   - **Total Items**: Number of nested parts
   - **Nested Layout**: Visual arrangement of parts
   - **Position Table**: Exact coordinates of each part
8. Download nested DXF file and nesting report

**Use Cases**:
- Production planning and optimization
- Material waste reduction
- Batch processing setup
- Cost optimization through efficient nesting

### 4. Image-to-DXF Conversion
**Purpose**: Convert raster images to cutting-ready DXF files

**Steps**:
1. Click **"2. Image to DXF"** from the main menu
2. Upload your image file (PNG, JPG, JPEG)
3. Adjust conversion parameters:
   - **Edge Threshold**: Sensitivity for edge detection (50-200)
   - **Canny Low/High**: Edge detection thresholds
   - **Minimum Contour Area**: Smallest feature size
   - **Simplify Tolerance**: Geometry simplification level
   - **Blur Kernel Size**: Image preprocessing blur
4. Click **"Convert Image"**
5. Review conversion results:
   - **Original Image**: Your uploaded image
   - **Edge Detection**: Detected edges preview
   - **DXF Preview**: Generated DXF visualization
   - **Quality Metrics**: Conversion quality scores
6. Download the generated DXF file
7. Proceed to DXF analysis or nesting

**Use Cases**:
- Converting logos and artwork to cutting files
- Creating DXF files from scanned drawings
- Batch processing of image files
- Rapid prototyping from images

### 5. Flooring Calculator
**Purpose**: Calculate tile layouts and costs for flooring projects

**Steps**:
1. Click **"Flooring Calculator"** from quick actions
2. Enter room dimensions:
   - **Room Width**: Room width in mm
   - **Room Length**: Room length in mm
3. Enter tile specifications:
   - **Tile Size**: Tile dimensions in mm
4. Click **"Calculate"**
5. Review results:
   - **Grid Layout**: Optimal tile arrangement
   - **Total Tiles**: Number of tiles needed
   - **Waste Calculation**: Material waste analysis
   - **Border Length**: Perimeter border requirements
   - **Cost Estimate**: Total project cost

**Use Cases**:
- Flooring project planning
- Material quantity estimation
- Cost calculation for quotes
- Layout optimization

## ⚙️ Configuration Options

### AI Model Selection
- **Auto Mode**: Automatically tries models in order of preference
- **Ollama Models**: Local AI models (no internet required)
  - `waterjet:latest`: Custom waterjet manufacturing model
  - `gpt-oss:20b`: General purpose model
  - `llama3.2-vision:latest`: Vision-capable model
- **OpenAI**: Cloud-based GPT-4 (requires API key)

### Analysis Parameters
- **Material Types**: Steel, aluminum, stainless steel, titanium
- **Thickness Range**: 0.5mm to 100mm
- **Kerf Width**: 0.05mm to 1.0mm
- **Cutting Speeds**: 100-3000 mm/min
- **Cost Rates**: Customizable per meter pricing

### Nesting Options
- **Sheet Sizes**: Standard sizes (1500x3000mm, 2000x4000mm) or custom
- **Spacing**: 5-50mm between parts
- **Algorithm**: Simple row-based nesting (advanced algorithms planned)

## 📊 Understanding Results

### DXF Analysis Results
- **Cutting Length**: Total distance the waterjet will travel
- **Pierce Points**: Number of times the jet needs to pierce the material
- **Estimated Time**: Calculated cutting time based on speeds
- **Cost Breakdown**: Material cost + cutting time cost + setup cost
- **Quality Issues**: Manufacturing concerns that need attention
- **Warnings**: Non-critical issues that should be reviewed

### AI Analysis Results
- **Feasibility Score**: 0-100 rating of manufacturing difficulty
  - 90-100: Excellent, easy to manufacture
  - 70-89: Good, minor considerations needed
  - 50-69: Moderate, some challenges expected
  - 0-49: Difficult, significant challenges likely
- **Complexity Level**: Simple/Moderate/Complex classification
- **Recommendations**: Specific suggestions for improvement

### Nesting Results
- **Utilization Percentage**: How much of the sheet is used
- **Part Positions**: Exact X,Y coordinates for each part
- **Row Information**: Which row each part is placed in
- **Waste Analysis**: Amount of unused material

## 🔧 Troubleshooting

### Common Issues

**"File doesn't fit on sheet" Warning**
- **Cause**: Part is larger than the specified sheet size
- **Solution**: Increase sheet dimensions or reduce part size

**"AI analysis failed" Error**
- **Cause**: AI service unavailable or timeout
- **Solution**: Try a different AI model or check internet connection

**"No module named 'ezdxf'" Error**
- **Cause**: Missing Python dependencies
- **Solution**: Run `pip install -r requirements.txt`

**"Potrace is not installed or not found in PATH" Error**
- **Cause**: Potrace executable not installed or not in system PATH
- **Solution**: Install Potrace using platform-specific instructions:
  - **Windows**: `choco install potrace` or download from http://potrace.sourceforge.net/
  - **Linux**: `sudo apt-get install potrace` (Ubuntu/Debian) or `sudo yum install potrace` (RHEL/CentOS)
  - **macOS**: `brew install potrace`

**Template Errors**
- **Cause**: Browser caching old templates
- **Solution**: Hard refresh browser (Ctrl+F5) or clear cache

### Performance Tips

**For Large Files**:
- Use smaller timeout values for faster processing
- Consider breaking large files into smaller parts
- Ensure sufficient system memory (8GB+ recommended)

**For Better AI Results**:
- Use higher timeout values (5-10 minutes) for complex analysis
- Try different AI models if one fails
- Ensure stable internet connection for OpenAI models

**For Nesting Optimization**:
- Use appropriate sheet sizes for your parts
- Adjust spacing based on material and cutting requirements
- Consider part orientation for better utilization

## 📞 Support & Resources

### Getting Help
- **Documentation**: Check the project documentation files
- **Error Messages**: Read error messages carefully for specific guidance
- **Log Files**: Check terminal output for detailed error information

### Best Practices
1. **File Preparation**: Ensure DXF files are clean and properly formatted
2. **Parameter Settings**: Use realistic cutting parameters for accurate results
3. **Quality Checks**: Always review quality issues and warnings
4. **Backup Files**: Keep copies of original files before processing
5. **Regular Updates**: Keep the system updated for best performance

### System Maintenance
- **Regular Restarts**: Restart the web server periodically
- **Clear Cache**: Clear browser cache if experiencing issues
- **Update Dependencies**: Keep Python packages updated
- **Monitor Resources**: Check system resources during heavy usage

---

**Manual Version**: 1.0.0  
**Last Updated**: September 28, 2025  
**For Technical Support**: Contact development team
