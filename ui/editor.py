# ui/editor.py - Advanced Constraint Editor Component
"""
Advanced Constitutional Constraint Editor with Templates and Validation
"""

import streamlit as st
import yaml
import json
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


class ConstraintTemplateLibrary:
    """Library of constraint templates for different use cases"""

    @staticmethod
    def get_template_categories() -> Dict[str, Dict[str, str]]:
        return {
            "Emotional Support": {
                "anger_management": """constraints:
  emotional_boundaries:
    - name: "anger_de_escalation"
      pattern: "angry|furious|rage|hate|livid"
      response: "acknowledge_emotion"
      severity: "high"
      template: "I can sense you're feeling really angry right now. That's a valid emotion. Let's work through this together - what's driving these feelings?"
      cooldown_minutes: 10
      max_triggers_per_day: 5""",

                "anxiety_support": """constraints:
  emotional_boundaries:
    - name: "anxiety_grounding"
      pattern: "anxious|worried|panic|overwhelmed|stressed"
      response: "acknowledge_emotion"
      severity: "medium"
      template: "I notice you're feeling anxious. Let's take a moment to ground ourselves. Can you name 3 things you can see around you?"
      techniques: ["grounding", "breathing"]""",

                "depression_care": """constraints:
  emotional_boundaries:
    - name: "depression_support"
      pattern: "depressed|hopeless|worthless|suicide|self.harm"
      response: "acknowledge_emotion"
      severity: "critical"
      template: "I'm concerned about how you're feeling. Your life has value. Would you like to talk about getting some support?"
      escalation_required: true
      professional_resources: true"""
            },

            "Privacy Protection": {
                "comprehensive_pii": """constraints:
  privacy_rules:
    - name: "email_protection"
      condition: "contains_email == true"
      action: "anonymize"
      severity: "high"

    - name: "phone_protection"
      condition: "contains_phone == true"
      action: "anonymize"
      severity: "high"

    - name: "ssn_protection"
      condition: "contains_ssn == true"
      action: "block"
      severity: "critical"
      template: "I cannot process social security numbers for your protection."

    - name: "financial_data"
      condition: "contains_financial_info == true"
      action: "anonymize"
      severity: "high"
      template: "I'll help with financial questions but need to anonymize specific numbers."

    - name: "medical_privacy"
      condition: "contains_medical_info == true"
      action: "local_only"
      severity: "high"
      template: "Medical information will be processed locally only for your privacy."
      """,

                "gdpr_compliance": """constraints:
  privacy_rules:
    - name: "eu_resident_protection"
      condition: "user_location == 'EU'"
      action: "local_only"
      severity: "high"
      gdpr_compliance: true

    - name: "right_to_deletion"
      condition: "user_requests_deletion == true"
      action: "delete_all_data"
      severity: "high"
      immediate: true

    - name: "data_portability"
      condition: "user_requests_export == true"
      action: "export_user_data"
      severity: "medium"
      format: "json"
      encrypted: true"""
            },

            "Healthy Usage": {
                "session_limits": """constraints:
  interaction_limits:
    - name: "session_duration_warning"
      condition: "session_duration > 30"
      action: "suggest_break"
      severity: "low"
      template: "You've been chatting for a while. Taking breaks can be healthy for processing emotions."

    - name: "session_duration_limit"
      condition: "session_duration > 60"
      action: "suggest_break"
      severity: "medium"
      template: "It's been over an hour. I recommend taking some time to reflect on our conversation."

    - name: "daily_interaction_limit"
      condition: "daily_interactions > 20"
      action: "suggest_break"
      severity: "medium"
      template: "You've had many conversations today. Sometimes space can be as valuable as connection."
      """,

                "emotional_intensity": """constraints:
  interaction_limits:
    - name: "high_emotion_pause"
      condition: "emotional_intensity > 0.8"
      action: "suggest_break"
      severity: "high"
      template: "I can sense you're experiencing very intense emotions. Sometimes a brief pause can help process these feelings."

    - name: "emotional_spiral_prevention"
      condition: "negative_emotion_streak > 3"
      action: "suggest_break"
      severity: "high"
      template: "I notice we've focused on difficult emotions for a while. Would it help to take a break or try a different approach?"

    - name: "trauma_processing_limit"
      condition: "trauma_discussion_duration > 15"
      action: "suggest_professional_help"
      severity: "critical"
      template: "Processing trauma is important work that often benefits from professional support. I can help you find resources."
      """
            },

            "Workplace Safety": {
                "professional_boundaries": """constraints:
  behavioral_guidelines:
    - name: "workplace_appropriate"
      description: "Maintain professional communication standards"
      required: true
      validation: ["no_personal_relationships", "professional_tone"]

    - name: "confidentiality_protection"
      description: "Never store or repeat confidential business information"
      required: true
      validation: ["no_business_secrets", "no_client_data"]

  privacy_rules:
    - name: "work_email_protection"
      condition: "contains_work_email == true"
      action: "anonymize"
      severity: "high"

    - name: "company_data_local_only"
      condition: "contains_company_info == true"
      action: "local_only"
      severity: "critical"
      """
            }
        }

    @staticmethod
    def get_constraint_examples() -> Dict[str, str]:
        """Get individual constraint examples for learning"""
        return {
            "Basic Emotional Boundary": """- name: "sadness_support"
  pattern: "sad|down|depressed|blue"
  response: "acknowledge_emotion"
  severity: "medium"
  template: "I can hear that you're feeling down. Would you like to talk about what's contributing to these feelings?"
  """,

            "Privacy Rule": """- name: "location_privacy"
  condition: "contains_location == true"
  action: "anonymize"
  severity: "high"
  template: "I'll help with location-based questions but will anonymize specific addresses."
  """,

            "Interaction Limit": """- name: "late_night_concern"
  condition: "hour > 23 OR hour < 6"
  action: "suggest_break"
  severity: "low"
  template: "It's quite late. Getting rest is important for emotional well-being. We can continue this conversation tomorrow."
  """,

            "Behavioral Guideline": """- name: "therapeutic_boundaries"
  description: "Maintain appropriate therapeutic boundaries"
  required: true
  validation: ["no_diagnosis", "no_medication_advice", "encourage_professional_help"]
  """
        }


def validate_yaml_constraints(yaml_content: str) -> tuple[bool, str, Dict]:
    """Validate YAML constraints format and content"""
    try:
        # Parse YAML
        data = yaml.safe_load(yaml_content)

        if not isinstance(data, dict) or 'constraints' not in data:
            return False, "YAML must contain a 'constraints' key", {}

        constraints = data['constraints']

        # Basic structure validation
        valid_sections = ['emotional_boundaries', 'privacy_rules', 'interaction_limits', 'behavioral_guidelines']
        for section in constraints.keys():
            if section not in valid_sections:
                return False, f"Unknown constraint section: {section}", {}

        # Validate each constraint type
        for section_name, section_constraints in constraints.items():
            if not isinstance(section_constraints, list):
                return False, f"Section '{section_name}' must be a list", {}

            for constraint in section_constraints:
                if not isinstance(constraint, dict) or 'name' not in constraint:
                    return False, f"Each constraint in '{section_name}' must have a 'name' field", {}

        return True, "Valid YAML constraints", data

    except yaml.YAMLError as e:
        return False, f"YAML parsing error: {str(e)}", {}
    except Exception as e:
        return False, f"Validation error: {str(e)}", {}


def advanced_constraint_editor():
    """Advanced constraint editor with templates and validation"""

    st.markdown("### 🎨 Advanced Constraint Editor")

    # Template selection
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 📚 Template Library")

        # Category selection
        templates = ConstraintTemplateLibrary.get_template_categories()
        selected_category = st.selectbox(
            "Choose Category:",
            options=list(templates.keys()),
            key="template_category"
        )

        # Template selection within category
        if selected_category:
            template_options = list(templates[selected_category].keys())
            selected_template = st.selectbox(
                "Choose Template:",
                options=template_options,
                key="template_choice"
            )

            # Preview and load template
            if selected_template:
                st.markdown("**Preview:**")
                template_content = templates[selected_category][selected_template]
                st.code(template_content, language="yaml")

                if st.button("Load Template", key="load_template"):
                    st.session_state.constraint_editor_content = template_content
                    st.success("Template loaded!")

        # Examples section
        st.markdown("#### 💡 Examples")
        examples = ConstraintTemplateLibrary.get_constraint_examples()
        selected_example = st.selectbox(
            "View Example:",
            options=list(examples.keys()),
            key="example_choice"
        )

        if selected_example:
            st.code(examples[selected_example], language="yaml")

    with col2:
        st.markdown("#### ✏️ Constraint Editor")

        # Initialize editor content
        if 'constraint_editor_content' not in st.session_state:
            st.session_state.constraint_editor_content = """constraints:
  emotional_boundaries:
    - name: "example_boundary"
      pattern: "sad|upset|down"
      response: "acknowledge_emotion"
      severity: "medium"
      template: "I can see you're feeling upset. Would you like to talk about it?"

  privacy_rules:
    - name: "example_privacy"
      condition: "contains_email == true"
      action: "anonymize"
      severity: "high"
"""

        # YAML editor
        yaml_content = st.text_area(
            "Constitutional Constraints (YAML):",
            value=st.session_state.constraint_editor_content,
            height=400,
            key="yaml_editor",
            help="Write your constitutional constraints in YAML format"
        )

        # Update session state
        st.session_state.constraint_editor_content = yaml_content

        # Validation
        col_validate, col_save = st.columns(2)

        with col_validate:
            if st.button("🔍 Validate", key="validate_constraints"):
                is_valid, message, parsed_data = validate_yaml_constraints(yaml_content)

                if is_valid:
                    st.success(f"✅ {message}")

                    # Show constraint summary
                    st.markdown("**Constraint Summary:**")
                    for section, constraints in parsed_data['constraints'].items():
                        st.write(f"- **{section}**: {len(constraints)} constraint(s)")
                        for constraint in constraints:
                            st.write(f"  - {constraint['name']}")
                else:
                    st.error(f"❌ {message}")

        with col_save:
            if st.button("💾 Save Configuration", key="save_constraints"):
                is_valid, message, parsed_data = validate_yaml_constraints(yaml_content)

                if is_valid:
                    # Save to session state for use in main app
                    st.session_state.current_constraints = parsed_data
                    st.success("Configuration saved!")
                else:
                    st.error(f"Cannot save invalid configuration: {message}")

        # Advanced options
        with st.expander("🔧 Advanced Options"):
            st.markdown("**Import/Export:**")

            col_import, col_export = st.columns(2)

            with col_import:
                uploaded_file = st.file_uploader(
                    "Import YAML file:",
                    type=['yaml', 'yml'],
                    key="import_constraints"
                )

                if uploaded_file is not None:
                    try:
                        content = uploaded_file.read().decode('utf-8')
                        st.session_state.constraint_editor_content = content
                        st.success("File imported!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Import failed: {str(e)}")

            with col_export:
                if st.button("Export Current", key="export_constraints"):
                    st.download_button(
                        label="Download YAML",
                        data=yaml_content,
                        file_name="ghost_protocol_constraints.yaml",
                        mime="text/yaml"
                    )

            st.markdown("**Validation Settings:**")
            strict_validation = st.checkbox(
                "Strict validation mode",
                value=True,
                help="Enforce all constraint fields and types"
            )

            show_schema = st.checkbox(
                "Show JSON Schema",
                help="Display the constraint validation schema"
            )

            if show_schema:
                schema = {
                    "constraints": {
                        "emotional_boundaries": ["name", "pattern", "response", "severity"],
                        "privacy_rules": ["name", "condition", "action", "severity"],
                        "interaction_limits": ["name", "condition", "action", "severity"],
                        "behavioral_guidelines": ["name", "description", "required"]
                    }
                }
                st.json(schema)


def constraint_builder_wizard():
    """Guided wizard for building constraints step by step"""

    st.markdown("### 🧙‍♂️ Constraint Builder Wizard")
    st.markdown("Build constraints step-by-step with guided assistance.")

    # Wizard state
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
    if 'wizard_constraint' not in st.session_state:
        st.session_state.wizard_constraint = {}

    # Progress indicator
    progress = st.session_state.wizard_step / 5
    st.progress(progress)
    st.write(f"Step {st.session_state.wizard_step} of 5")

    if st.session_state.wizard_step == 1:
        st.markdown("#### Step 1: Constraint Type")
        constraint_type = st.radio(
            "What type of constraint do you want to create?",
            options=[
                "Emotional Boundary - Respond to emotional states",
                "Privacy Rule - Protect sensitive information",
                "Interaction Limit - Manage usage patterns",
                "Behavioral Guideline - Define system behavior"
            ],
            key="wizard_constraint_type"
        )

        st.session_state.wizard_constraint['type'] = constraint_type.split(' - ')[0].lower().replace(' ', '_')

        if st.button("Next →", key="wizard_step1_next"):
            st.session_state.wizard_step = 2
            st.experimental_rerun()

    elif st.session_state.wizard_step == 2:
        st.markdown("#### Step 2: Basic Information")

        constraint_name = st.text_input(
            "Constraint Name:",
            placeholder="e.g., anger_support, email_privacy",
            key="wizard_constraint_name"
        )

        constraint_description = st.text_area(
            "Description:",
            placeholder="Describe what this constraint does...",
            key="wizard_constraint_description"
        )

        st.session_state.wizard_constraint['name'] = constraint_name
        st.session_state.wizard_constraint['description'] = constraint_description

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Back", key="wizard_step2_back"):
                st.session_state.wizard_step = 1
                st.experimental_rerun()
        with col_next:
            if st.button("Next →", key="wizard_step2_next") and constraint_name:
                st.session_state.wizard_step = 3
                st.experimental_rerun()

    elif st.session_state.wizard_step == 3:
        st.markdown("#### Step 3: Trigger Conditions")

        constraint_type = st.session_state.wizard_constraint['type']

        if constraint_type == 'emotional_boundary':
            pattern = st.text_input(
                "Emotion Pattern (regex):",
                placeholder="sad|upset|down|depressed",
                help="Regular expression to match emotional words",
                key="wizard_pattern"
            )
            st.session_state.wizard_constraint['pattern'] = pattern

        elif constraint_type == 'privacy_rule':
            condition = st.selectbox(
                "Privacy Condition:",
                options=[
                    "contains_email == true",
                    "contains_phone == true",
                    "contains_ssn == true",
                    "contains_address == true",
                    "contains_financial_info == true"
                ],
                key="wizard_privacy_condition"
            )
            st.session_state.wizard_constraint['condition'] = condition

        elif constraint_type == 'interaction_limit':
            condition = st.selectbox(
                "Limit Condition:",
                options=[
                    "session_duration > 30",
                    "daily_interactions > 10",
                    "emotional_intensity > 0.8",
                    "hour > 23 OR hour < 6"
                ],
                key="wizard_limit_condition"
            )
            st.session_state.wizard_constraint['condition'] = condition

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Back", key="wizard_step3_back"):
                st.session_state.wizard_step = 2
                st.experimental_rerun()
        with col_next:
            if st.button("Next →", key="wizard_step3_next"):
                st.session_state.wizard_step = 4
                st.experimental_rerun()

    elif st.session_state.wizard_step == 4:
        st.markdown("#### Step 4: Response Configuration")

        severity = st.selectbox(
            "Severity Level:",
            options=["low", "medium", "high", "critical"],
            key="wizard_severity"
        )

        constraint_type = st.session_state.wizard_constraint['type']

        if constraint_type == 'emotional_boundary':
            response_type = st.selectbox(
                "Response Type:",
                options=["acknowledge_emotion", "redirect_conversation", "suggest_break"],
                key="wizard_response_type"
            )
            st.session_state.wizard_constraint['response'] = response_type

        elif constraint_type in ['privacy_rule', 'interaction_limit']:
            action = st.selectbox(
                "Action:",
                options=["anonymize", "block", "local_only", "suggest_break"],
                key="wizard_action"
            )
            st.session_state.wizard_constraint['action'] = action

        template = st.text_area(
            "Response Template:",
            placeholder="How should the AI respond when this constraint is triggered?",
            key="wizard_template"
        )

        st.session_state.wizard_constraint['severity'] = severity
        st.session_state.wizard_constraint['template'] = template

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Back", key="wizard_step4_back"):
                st.session_state.wizard_step = 3
                st.experimental_rerun()
        with col_next:
            if st.button("Next →", key="wizard_step4_next"):
                st.session_state.wizard_step = 5
                st.experimental_rerun()

    elif st.session_state.wizard_step == 5:
        st.markdown("#### Step 5: Review & Generate")

        st.markdown("**Constraint Summary:**")
        constraint = st.session_state.wizard_constraint

        for key, value in constraint.items():
            if value:
                st.write(f"- **{key.title()}**: {value}")

        # Generate YAML
        if st.button("Generate YAML", key="wizard_generate"):
            constraint_type = constraint['type']
            yaml_constraint = generate_constraint_yaml(constraint)

            st.markdown("**Generated YAML:**")
            st.code(yaml_constraint, language="yaml")

            if st.button("Add to Editor", key="wizard_add_to_editor"):
                current_content = st.session_state.get('constraint_editor_content', 'constraints:\n')
                st.session_state.constraint_editor_content = current_content + "\n" + yaml_constraint
                st.success("Constraint added to editor!")

        col_back, col_restart = st.columns(2)
        with col_back:
            if st.button("← Back", key="wizard_step5_back"):
                st.session_state.wizard_step = 4
                st.experimental_rerun()
        with col_restart:
            if st.button("🔄 New Constraint", key="wizard_restart"):
                st.session_state.wizard_step = 1
                st.session_state.wizard_constraint = {}
                st.experimental_rerun()


def generate_constraint_yaml(constraint: Dict[str, Any]) -> str:
    """Generate YAML for a constraint from wizard data"""

    constraint_type = constraint['type']

    if constraint_type == 'emotional_boundary':
        yaml_content = f"""  emotional_boundaries:
    - name: "{constraint['name']}"
      pattern: "{constraint.get('pattern', '')}"
      response: "{constraint.get('response', 'acknowledge_emotion')}"
      severity: "{constraint.get('severity', 'medium')}"
      template: "{constraint.get('template', '')}" """

    elif constraint_type == 'privacy_rule':
        yaml_content = f"""  privacy_rules:
    - name: "{constraint['name']}"
      condition: "{constraint.get('condition', '')}"
      action: "{constraint.get('action', 'anonymize')}"
      severity: "{constraint.get('severity', 'high')}"
      template: "{constraint.get('template', '')}" """

    elif constraint_type == 'interaction_limit':
        yaml_content = f"""  interaction_limits:
    - name: "{constraint['name']}"
      condition: "{constraint.get('condition', '')}"
      action: "{constraint.get('action', 'suggest_break')}"
      severity: "{constraint.get('severity', 'medium')}"
      template: "{constraint.get('template', '')}" """

    elif constraint_type == 'behavioral_guideline':
        yaml_content = f"""  behavioral_guidelines:
    - name: "{constraint['name']}"
      description: "{constraint.get('description', '')}"
      required: true
      validation: ["custom_validation"] """

    else:
        yaml_content = f"""  # Generated constraint for {constraint['name']}
    - name: "{constraint['name']}"
      # Add constraint details here """

    return yaml_content


def main():
    """Main function for testing the editor component"""
    st.set_page_config(
        page_title="Ghost Protocol - Constraint Editor",
        page_icon="🎨",
        layout="wide"
    )

    st.title("🎨 Ghost Protocol Constraint Editor")

    tab1, tab2 = st.tabs(["Advanced Editor", "Builder Wizard"])

    with tab1:
        advanced_constraint_editor()

    with tab2:
        constraint_builder_wizard()


if __name__ == "__main__":
    main()