import pydicom
from pydicom.sr.coding import Code
from pydicom.sr.codedict import codes
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from datetime import datetime

class DicomSRGenerator:
    def __init__(self):
        self.sr_dataset = Dataset()
        
    def create_amyloid_sr(self, centiloid_score, aal_suvr_values, description, study_info):
        """Create DICOM SR for Amyloid PET analysis."""
        # Basic SR attributes
        self._add_sr_metadata(study_info)
        
        # Create content sequence
        content_seq = Sequence()
        
        # Add Centiloid score
        content_seq.append(self._create_numeric_measurement(
            name="Centiloid Score",
            value=centiloid_score,
            unit="CL",
            code="126701"
        ))
        
        # Add regional SUVr values
        for region, value in aal_suvr_values.items():
            content_seq.append(self._create_numeric_measurement(
                name=f"SUVr {region}",
                value=value,
                unit="ratio",
                code="126702"
            ))
        
        # Add findings
        content_seq.append(self._create_text_content(
            name="Findings",
            value=description,
            code="121071"
        ))
        
        self.sr_dataset.ContentSequence = content_seq
        return self.sr_dataset
    
    def _add_sr_metadata(self, study_info):
        """Add DICOM SR metadata."""
        ds = self.sr_dataset
        
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.33"  # Comprehensive SR
        ds.Modality = "SR"
        ds.ContentDate = datetime.now().strftime("%Y%m%d")
        ds.ContentTime = datetime.now().strftime("%H%M%S")
        
        for key, value in study_info.items():
            setattr(ds, key, value)