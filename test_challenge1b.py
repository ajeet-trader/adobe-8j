"""
Test and validation script for Challenge 1B solution
"""

import json
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Challenge1BValidator:
    def __init__(self):
        self.required_output_schema = {
            "metadata": ["input_documents", "persona", "job_to_be_done"],
            "extracted_sections": ["document", "section_title", "importance_rank", "page_number"],
            "subsection_analysis": ["document", "refined_text", "page_number"]
        }
    
    def validate_output(self, output_file: Path) -> Dict[str, Any]:
        """Validate the output JSON against expected schema"""
        validation_results = {
            "file_exists": False,
            "valid_json": False,
            "schema_valid": False,
            "content_quality": {},
            "errors": []
        }
        
        # Check if file exists
        if not output_file.exists():
            validation_results["errors"].append(f"Output file not found: {output_file}")
            return validation_results
        
        validation_results["file_exists"] = True
        
        # Check if valid JSON
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            validation_results["valid_json"] = True
        except json.JSONDecodeError as e:
            validation_results["errors"].append(f"Invalid JSON: {e}")
            return validation_results
        
        # Validate schema
        schema_errors = self._validate_schema(data)
        if not schema_errors:
            validation_results["schema_valid"] = True
        else:
            validation_results["errors"].extend(schema_errors)
        
        # Validate content quality
        validation_results["content_quality"] = self._validate_content_quality(data)
        
        return validation_results
    
    def _validate_schema(self, data: Dict) -> List[str]:
        """Validate JSON schema structure"""
        errors = []
        
        # Check top-level keys
        required_keys = ["metadata", "extracted_sections", "subsection_analysis"]
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required key: {key}")
        
        # Check metadata structure
        if "metadata" in data:
            for required_field in self.required_output_schema["metadata"]:
                if required_field not in data["metadata"]:
                    errors.append(f"Missing metadata field: {required_field}")
        
        # Check extracted_sections structure
        if "extracted_sections" in data and data["extracted_sections"]:
            for i, section in enumerate(data["extracted_sections"][:5]):  # Check first 5
                for required_field in self.required_output_schema["extracted_sections"]:
                    if required_field not in section:
                        errors.append(f"Missing field in extracted_sections[{i}]: {required_field}")
        
        # Check subsection_analysis structure
        if "subsection_analysis" in data and data["subsection_analysis"]:
            for i, subsection in enumerate(data["subsection_analysis"][:5]):  # Check first 5
                for required_field in self.required_output_schema["subsection_analysis"]:
                    if required_field not in subsection:
                        errors.append(f"Missing field in subsection_analysis[{i}]: {required_field}")
        
        return errors
    
    def _validate_content_quality(self, data: Dict) -> Dict[str, Any]:
        """Validate content quality metrics"""
        quality_metrics = {
            "total_sections": len(data.get("extracted_sections", [])),
            "total_subsections": len(data.get("subsection_analysis", [])),
            "unique_documents": len(set(s.get("document", "") for s in data.get("extracted_sections", []))),
            "ranking_valid": True,
            "content_length_avg": 0,
            "persona_relevance": "unknown"
        }
        
        # Check ranking validity
        sections = data.get("extracted_sections", [])
        if sections:
            ranks = [s.get("importance_rank", 0) for s in sections]
            expected_ranks = list(range(1, len(sections) + 1))
            if sorted(ranks) != expected_ranks:
                quality_metrics["ranking_valid"] = False
        
        # Calculate average content length
        subsections = data.get("subsection_analysis", [])
        if subsections:
            total_length = sum(len(s.get("refined_text", "")) for s in subsections)
            quality_metrics["content_length_avg"] = total_length / len(subsections)
        
        # Determine persona relevance (basic check)
        persona = data.get("metadata", {}).get("persona", "")
        if persona:
            quality_metrics["persona_relevance"] = persona
        
        return quality_metrics

def run_validation():
    """Run validation on all collection outputs"""
    validator = Challenge1BValidator()
    base_path = Path(".")
    
    collections = ["Collection 1", "Collection 2", "Collection 3"]
    
    logger.info("Starting Challenge 1B Output Validation")
    logger.info("=" * 50)
    
    all_valid = True
    
    for collection in collections:
        output_file = base_path / collection / "challenge1b_output.json"
        
        logger.info(f"\nValidating {collection}...")
        results = validator.validate_output(output_file)
        
        if results["file_exists"] and results["valid_json"] and results["schema_valid"]:
            logger.info(f"✅ {collection}: PASSED")
            
            # Print quality metrics
            quality = results["content_quality"]
            logger.info(f"   Sections: {quality['total_sections']}")
            logger.info(f"   Subsections: {quality['total_subsections']}")
            logger.info(f"   Documents covered: {quality['unique_documents']}")
            logger.info(f"   Ranking valid: {quality['ranking_valid']}")
            logger.info(f"   Avg content length: {quality['content_length_avg']:.1f} chars")
            
        else:
            logger.error(f"❌ {collection}: FAILED")
            for error in results["errors"]:
                logger.error(f"   - {error}")
            all_valid = False
    
    logger.info("\n" + "=" * 50)
    if all_valid:
        logger.info("🎉 ALL VALIDATIONS PASSED!")
    else:
        logger.error("❌ Some validations failed. Please check the errors above.")
    
    return all_valid

if __name__ == "__main__":
    run_validation()
