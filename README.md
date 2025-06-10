### Oracle Bone Multimodal Dataset (OBIMD)  
**Overview**  
OBIMD is a pioneering multimodal dataset for **oracle bone script research**, combining visual, textual, and structural data. It bridges ancient Chinese writing systems with modern computational analysis.

**Key Components**  
1. **Images**: High-resolution photos/scans of oracle bone fragments.  
2. **Annotations**:  
   - Character-level bounding boxes  
   - Modern Chinese transcriptions  
   - Phonetic notations  
   - Semantic categories  
   - Stroke-order labels  
3. **Metadata**: Archaeological details (dating, excavation sites).  

**Applications**  
- Character recognition & detection  
- Paleographic studies  
- Multimodal AI training (vision + language)  
- Historical linguistics research  

**Dataset Stats**  
- ~5,000 annotated oracle bone images  
- Covers 3,000+ distinct ancient characters  

**Access**  
```python
from datasets import load_dataset
dataset = load_dataset("libang1991/OBIMD")
```

**Significance**  
Enables AI-driven analysis of the world’s oldest Chinese writing system (3,200+ years old), accelerating decipherment and preservation efforts.

**License**  
CC BY-NC-SA 4.0 (Non-commercial research use)

---

This refined summary highlights OBIMD’s role as a foundational resource for computational archaeology and multimodal AI research. The cleaned version removes redundant details while emphasizing its unique value for ancient script analysis.