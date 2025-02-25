import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from jinja2 import Template
from ..utils.llm import get_llm_description

class AmyloidReportGenerator:
    def __init__(self, config=None):
        self.config = config or Config()
        self.template_loader = self._initialize_template()
        
    def generate_report(self, 
                       pet_data, 
                       centiloid_score, 
                       aal_suvr_values, 
                       output_dir,
                       subject_info=None):
        """Generate comprehensive Amyloid PET report."""
        
        # Create report directory
        report_dir = Path(output_dir) / 'report'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate AAL SUVr table
        suvr_table = self._create_suvr_table(aal_suvr_values)
        
        # Get image description from LLM
        description = get_llm_description(
            centiloid_score=centiloid_score,
            suvr_values=aal_suvr_values,
            model="deepseek-coder:latest"
        )
        
        # Prepare report context
        context = {
            'subject_info': subject_info,
            'centiloid_score': centiloid_score,
            'suvr_table': suvr_table,
            'description': description,
            'date': datetime.now().strftime("%Y-%m-%d"),
        }
        
        # Generate report in multiple formats
        self._generate_pdf_report(context, report_dir)
        self._generate_json_report(context, report_dir)
        
        return str(report_dir / 'report.pdf')