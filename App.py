import json
import time
import re
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
# from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException

# YAML file configurations
YAML_CONFIGS = {
    "Salesforce": "SALESFORCEDB.PUBLIC.SALESFORCE_STAGE/salesforceyaml.yaml"
}

# Salesforce date/timestamp field patterns
# SALESFORCE_DATE_FIELDS = [
#     'END_DATE', 'START_DATE', 'CLOSE_DATE', 'CREATED_DATE', 'LAST_MODIFIED_DATE',
#     'LAST_ACTIVITY_DATE', 'LAST_VIEWED_DATE', 'LAST_REFERENCED_DATE',
#     'SYSTEM_MODSTAMP', 'BIRTHDAY', 'DUE_DATE', 'REMIND_DATE', 'ACTIVITY_DATE',
#     'COMPLETION_DATE', 'EXPIRATION_DATE', 'EFFECTIVE_DATE', 'ENROLLMENT_DATE'
# ]

# Pagination settings
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000
LARGE_DATASET_THRESHOLD = 1000  # When to automatically enable pagination

cnx = st.connection("snowflake")
session = cnx.session()

def convert_salesforce_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Salesforce timestamp fields from Unix milliseconds to readable dates.
    Handles both integer timestamps and None values.
    """
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Check if column name indicates it's a date/timestamp field
        col_upper = col.upper()
        if any(date_field in col_upper for date_field in SALESFORCE_DATE_FIELDS):
            try:
                # Convert the column, handling None values and non-numeric data
                def convert_timestamp(value):
                    if pd.isna(value) or value is None or value == 'None':
                        return None
                    
                    # Try to convert to integer (handle string numbers)
                    try:
                        timestamp_ms = int(float(value))
                        # Check if it's a reasonable timestamp (between 1970 and 2100)
                        if 0 < timestamp_ms < 4102444800000:  # Jan 1, 2100 in milliseconds
                            # Convert from milliseconds to seconds for datetime
                            timestamp_s = timestamp_ms / 1000
                            return datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S')
                        else:
                            return value  # Return original if not a valid timestamp
                    except (ValueError, TypeError, OSError):
                        return value  # Return original if conversion fails
                
                df_copy[col] = df_copy[col].apply(convert_timestamp)
                
            except Exception as e:
                # If there's any error with column conversion, leave it as is
                st.warning(f"Could not convert timestamps in column '{col}': {str(e)}")
                continue
    
    return df_copy


def detect_and_convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and convert potential timestamp columns even if they don't match known field names.
    This is a more comprehensive approach that looks at the data patterns.
    """
    if df is None or df.empty:
        return df
    
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Skip if already converted by name-based conversion
        col_upper = col.upper()
        if any(date_field in col_upper for date_field in SALESFORCE_DATE_FIELDS):
            continue
            
        # Check if column contains timestamp-like values
        sample_values = df_copy[col].dropna().head(10)
        if sample_values.empty:
            continue
            
        # Check if values look like Unix timestamps (13 digits for milliseconds)
        timestamp_pattern = re.compile(r'^\d{13}$')
        timestamp_count = sum(1 for val in sample_values if 
                            isinstance(val, (int, float, str)) and 
                            timestamp_pattern.match(str(val)))
        
        # If more than 70% of sample values look like timestamps, convert the column
        if timestamp_count / len(sample_values) > 0.7:
            try:
                def convert_detected_timestamp(value):
                    if pd.isna(value) or value is None or value == 'None':
                        return None
                    
                    try:
                        timestamp_ms = int(float(value))
                        if 1000000000000 <= timestamp_ms <= 4102444800000:  # Reasonable range
                            timestamp_s = timestamp_ms / 1000
                            return datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d')
                        else:
                            return value
                    except (ValueError, TypeError, OSError):
                        return value
                
                df_copy[col] = df_copy[col].apply(convert_detected_timestamp)
                st.info(f"🕒 Detected and converted timestamp column: '{col}'")
                
            except Exception as e:
                continue
    
    return df_copy


def split_dataframe(input_df: pd.DataFrame, rows: int) -> List[pd.DataFrame]:
    """
    Split dataframe into chunks for pagination.
    
    Args:
        input_df (pd.DataFrame): Input dataframe
        rows (int): Number of rows per chunk
        
    Returns:
        List[pd.DataFrame]: List of dataframe chunks
    """
    df_chunks = [input_df.iloc[i:i + rows] for i in range(0, len(input_df), rows)]
    return df_chunks


def display_pagination_controls(query_key: str, total_records: int, page_size: int, current_page: int):
    """
    Display improved pagination controls with proper state updates.
    FIXED: Proper state management and immediate updates.
    """
    total_pages = math.ceil(total_records / page_size) if total_records > 0 else 1
    
    if total_pages <= 1:
        return current_page
    
    # Track if any changes occurred
    state_changed = False
    new_page = current_page
    
    # Create a well-designed pagination container
    with st.container():
        # Info section with better formatting
        info_col, settings_col = st.columns([3, 1])
        
        with info_col:
            start_record = (current_page - 1) * page_size + 1
            end_record = min(current_page * page_size, total_records)
            st.markdown(f"""
            **📊 Results Overview**  
            Showing **{start_record:,} - {end_record:,}** of **{total_records:,}** records  
            Page **{current_page}** of **{total_pages}**
            """)
        
        with settings_col:
            # Page size selector with immediate effect
            current_page_size = st.session_state.get(f"page_size_{query_key}", page_size)
            new_page_size = st.selectbox(
                "📄 Per Page", 
                options=[25, 50, 100, 200, 500, 1000],
                index=[25, 50, 100, 200, 500, 1000].index(current_page_size) if current_page_size in [25, 50, 100, 200, 500, 1000] else 2,
                key=f"page_size_selector_{query_key}",
                help="Number of records to show per page"
            )
            
            # Handle page size change
            if new_page_size != current_page_size:
                st.session_state[f"page_size_{query_key}"] = new_page_size
                st.session_state[f"current_page_{query_key}"] = 1  # Reset to first page
                st.rerun()
        
        st.divider()
        
        # Navigation controls
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])
        
        with nav_col1:
            # First page button
            if st.button(⏮️ First", key=f"first_{query_key}", disabled=current_page <= 1, help="Go to first page"):
                new_page = 1
                state_changed = True
        
        with nav_col2:
            # Previous page button
            if st.button("⬅️ Prev", key=f"prev_{query_key}", disabled=current_page <= 1, help="Go to previous page"):
                new_page = current_page - 1
                state_changed = True
        
        with nav_col3:
            # Page number input
            page_input = st.number_input(
                "🔢 Jump to page",
                min_value=1,
                max_value=total_pages,
                value=current_page,
                key=f"page_input_{query_key}",
                help=f"Enter page number (1-{total_pages})"
            )
            if page_input != current_page:
                new_page = page_input
                state_changed = True
        
        with nav_col4:
            # Next page button
            if st.button("➡️ Next", key=f"next_{query_key}", disabled=current_page >= total_pages, help="Go to next page"):
                new_page = current_page + 1
                state_changed = True
        
        with nav_col5:
            # Last page button
            if st.button("⏭️ Last", key=f"last_{query_key}", disabled=current_page >= total_pages, help="Go to last page"):
                new_page = total_pages
                state_changed = True
        
        # Progress bar
        progress_value = current_page / total_pages
        st.progress(progress_value, text=f"Page {current_page} of {total_pages}")
        
        # Quick page jumps for large datasets
        if total_pages > 10:
            st.markdown("**Quick Jump:**")
            quick_jump_cols = st.columns(min(5, total_pages))
            
            # Show strategic page numbers
            quick_pages = []
            if total_pages <= 5:
                quick_pages = list(range(1, total_pages + 1))
            else:
                quick_pages = [1]
                if total_pages > 20:
                    step = max(1, total_pages // 4)
                    quick_pages.extend([step, step * 2, step * 3])
                elif total_pages > 10:
                    mid = total_pages // 2
                    quick_pages.extend([max(1, mid - 1), mid, min(total_pages, mid + 1)])
                quick_pages.append(total_pages)
                quick_pages = sorted(list(set(quick_pages)))
            
            for i, page_num in enumerate(quick_pages[:5]):
                if i < len(quick_jump_cols):
                    with quick_jump_cols[i]:
                        is_current = page_num == current_page
                        button_label = f"{'📍 ' if is_current else ''}{page_num}"
                        if st.button(
                            button_label, 
                            key=f"quick_{query_key}_{page_num}",
                            disabled=is_current,
                            help=f"Jump to page {page_num}"
                        ):
                            new_page = page_num
                            state_changed = True
    
    # Update state if changed
    if state_changed and new_page != current_page:
        st.session_state[f"current_page_{query_key}"] = new_page
        st.rerun()
    
    return new_page if state_changed else current_page

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        reset_session_state()
    if "selected_yaml" not in st.session_state:
        st.session_state.selected_yaml = "Salesforce"
    
    show_header_and_sidebar()
    
    # Show initial question only once
    if len(st.session_state.messages) == 0 and st.session_state.selected_yaml and "initial_question_asked" not in st.session_state:
        st.session_state.initial_question_asked = True
        process_user_input("What questions can I ask?")
    
    display_conversation()
    handle_user_inputs()
    handle_error_notifications()


def reset_session_state():
    """Reset important session state elements but preserve default settings."""
    st.session_state.messages = []
    st.session_state.active_suggestion = None
    st.session_state.warnings = []
    st.session_state.query_results_cache = {}  # Cache for query results
    # Clear all pagination states so new queries use current default
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith(('current_page_', 'page_size_', 'total_records_'))]
    for key in keys_to_remove:
        del st.session_state[key]
    # Keep default_page_size and page_size_setting
    if "initial_question_asked" in st.session_state:
        del st.session_state.initial_question_asked


def show_header_and_sidebar():
    """Display the header and sidebar of the app."""
    st.title("NLP-Based Dashboards")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Welcome to AI Analyst! Select your data source and ask questions about your data. Large datasets will automatically use pagination for better performance.")
    
    with col2:
        new_yaml_selection = st.selectbox(
            "Select Data Source:",
            options=list(YAML_CONFIGS.keys()),
            index=list(YAML_CONFIGS.keys()).index(st.session_state.selected_yaml),
            key="yaml_selector"
        )
        
        # Handle data source change
        if new_yaml_selection != st.session_state.selected_yaml:
            st.session_state.messages = []
            st.session_state.active_suggestion = None
            st.session_state.warnings = []
            st.session_state.selected_yaml = new_yaml_selection
            if "initial_question_asked" in st.session_state:
                del st.session_state.initial_question_asked
    
    # Sidebar with pagination settings
    with st.sidebar:
        st.subheader("Pagination Settings")
        
        # Get current default page size or set initial value
        current_default = st.session_state.get('default_page_size', 100)
        
        default_page_size = st.selectbox(
            "Default Page Size:",
            options=[25, 50, 100, 200, 500, 1000],
            index=[25, 50, 100, 200, 500, 1000].index(current_default) if current_default in [25, 50, 100, 200, 500, 1000] else 2,
            key="page_size_setting",
            help="This will be the default page size for new queries"
        )
        
        # Store the default page size in session state
        if default_page_size != st.session_state.get('default_page_size'):
            st.session_state['default_page_size'] = default_page_size
            # Show info message when changed
            st.info(f"✅ Default page size updated to {default_page_size}. This will apply to new queries.")
        
        st.divider()
        if st.button("Clear Chat History", use_container_width=True):
            reset_session_state()
    
    st.info(f"📊 **{st.session_state.selected_yaml}** data source")
    st.divider()


def handle_user_inputs():
    """Handle user inputs from the chat interface."""
    if not st.session_state.selected_yaml:
        st.warning("Please select a data source first.")
        return
    
    user_input = st.chat_input("What is your question?")
    if user_input:
        process_user_input(user_input)
    elif st.session_state.active_suggestion is not None:
        suggestion = st.session_state.active_suggestion
        st.session_state.active_suggestion = None
        process_user_input(suggestion)


def handle_error_notifications():
    """Handle error notifications."""
    if st.session_state.get("fire_API_error_notify"):
        st.toast("An API error has occurred!", icon="🚨")
        st.session_state["fire_API_error_notify"] = False


def process_user_input(prompt: str):
    """Process user input and update the conversation history."""
    # Clear previous warnings
    st.session_state.warnings = []

    # Create user message (hidden from UI)
    new_user_message = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
        "hidden": True
    }
    st.session_state.messages.append(new_user_message)
    
    # Prepare messages for API
    messages_for_api = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]

    # Show analyst response with progress
    with st.chat_message("analyst"):
        with st.spinner("🤔 Analyzing your Data..."):
            response, error_msg = get_analyst_response(messages_for_api)
            
            if error_msg is None:
                analyst_message = {
                    "role": "analyst",
                    "content": response["message"]["content"],
                    "request_id": response["request_id"],
                }
            else:
                analyst_message = {
                    "role": "analyst",
                    "content": [{"type": "text", "text": error_msg}],
                    "request_id": response.get("request_id", "error"),
                }
                st.session_state["fire_API_error_notify"] = True

            if "warnings" in response:
                st.session_state.warnings = response["warnings"]

            st.session_state.messages.append(analyst_message)
            st.rerun()


def display_warnings():
    """Display warnings to the user."""
    for warning in st.session_state.warnings:
        st.warning(warning["message"], icon="⚠️")


def get_analyst_response(messages: List[Dict]) -> Tuple[Dict, Optional[str]]:
    """
    Send chat history to the Cortex Analyst API via stored procedure.
    OPTIMIZED: Improved error handling and response processing.
    """
    selected_yaml_path = YAML_CONFIGS[st.session_state.selected_yaml]
    semantic_model_file = f"@{selected_yaml_path}"
    
    try:
        # Call stored procedure with timeout handling
        result = session.call(
            "CORTEX_ANALYST.CORTEX_AI.CORTEX_ANALYST_API_PROCEDURE",
            messages,
            semantic_model_file
        )
        
        if result is None:
            return {"request_id": "error"}, "❌ No response from Cortex Analyst procedure"
        
        # Parse response
        if isinstance(result, str):
            response_data = json.loads(result)
        else:
            response_data = result
        
        # Handle successful response
        if response_data.get("success", False):
            return_data = {
                "message": response_data.get("analyst_response", {}),
                "request_id": response_data.get("request_id", "N/A"),
                "warnings": response_data.get("warnings", [])
            }
            return return_data, None
        
        # Handle error response
        error_details = response_data.get("error_details", {})
        error_msg = f"""
❌ **Cortex Analyst Error**

**Error Code:** `{error_details.get('error_code', 'N/A')}`  
**Request ID:** `{error_details.get('request_id', 'N/A')}`  
**Status:** `{error_details.get('response_code', 'N/A')}`

**Message:** {error_details.get('error_message', 'No error message provided')}

💡 **Troubleshooting:**
- Verify your {st.session_state.selected_yaml.lower()}.yaml file exists in the stage
- Check database and schema permissions
- Ensure Cortex Analyst is properly configured
        """
        
        return_data = {
            "request_id": response_data.get("request_id", "error"),
            "warnings": response_data.get("warnings", [])
        }
        return return_data, error_msg
        
    except SnowparkSQLException as e:
        error_msg = f"""
❌ **Database Error**

{str(e)}

💡 **Check:**
- Procedure exists: `CORTEX_ANALYST.CORTEX_AI.CORTEX_ANALYST_API_PROCEDURE`
- You have EXECUTE permissions
- YAML file exists in stage
        """
        return {"request_id": "error"}, error_msg
        
    except Exception as e:
        error_msg = f"❌ **Unexpected Error:** {str(e)}"
        return {"request_id": "error"}, error_msg


def display_conversation():
    """Display the conversation history (excluding hidden messages)."""
    for idx, message in enumerate(st.session_state.messages):
        if message.get("hidden", False):
            continue
            
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            display_message(content, idx)


def display_message(content: List[Dict[str, Union[str, Dict]]], message_index: int):
    """Display a single message content."""
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
        elif item["type"] == "suggestions":
            st.markdown("**💡 Suggested questions:**")
            for suggestion_index, suggestion in enumerate(item["suggestions"]):
                if st.button(
                    suggestion, 
                    key=f"suggestion_{message_index}_{suggestion_index}",
                    type="secondary"
                ):
                    st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            display_sql_query(
                item["statement"], message_index, item.get("confidence")
            )


def modify_salesforce_query(sql: str) -> str:
    """
    Optimize SQL queries by removing 'public' schema from salesforceDb references.
    OPTIMIZED: More efficient regex processing.
    """
    import re
    
    # Single pass with multiple patterns
    patterns = [
        (r'("[sS][aA][lL][eE][sS][fF][oO][rR][cC][eE][dD][bB]")\.("[pP][uU][bB][lL][iI][cC]")\.', r'\1.'),
        (r'\b([sS][aA][lL][eE][sS][fF][oO][rR][cC][eE][dD][bB])\.([pP][uU][bB][lL][iI][cC])\.', r'\1.'),
        (r'("[sS][aA][lL][eE][sS][fF][oO][rR][cC][eE][dD][bB]")\.([pP][uU][bB][lL][iI][cC])\.', r'\1.'),
        (r'\b([sS][aA][lL][eE][sS][fF][oO][rR][cC][eE][dD][bB])\.("[pP][uU][bB][lL][iI][cC]")\.', r'\1.')
    ]
    
    for pattern, replacement in patterns:
        sql = re.sub(pattern, replacement, sql)
    
    return sql


# Replace your execute_data_procedure function with this enhanced version:

@st.cache_data(show_spinner=False, ttl=300)
def execute_data_procedure(query: str, data_source: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute data procedure with enhanced error handling and user guidance.
    """
    try:
        # All data sources use the unified Dremio procedure
        if data_source == "Salesforce":
            modified_query = modify_salesforce_query(query)
            procedure_call = f"CALL SALESFORCE_DREMIO.SALESFORCE_SCHEMA_DREMIO.dremio_data_procedure('{modified_query}')"
        elif data_source == "Odoo":
            procedure_call = f"CALL SALESFORCE_DREMIO.SALESFORCE_SCHEMA_DREMIO.dremio_data_procedure('{query}')"
        elif data_source == "SAP":
            procedure_call = f"CALL SALESFORCE_DREMIO.SALESFORCE_SCHEMA_DREMIO.dremio_data_procedure('{query}')"
        else:
            return None, f"❌ Unknown data source: {data_source}"
        
        # Execute the procedure
        result = session.sql(procedure_call)
        df = result.to_pandas()
        
        if df is None or df.empty:
            return None, "📭 No data available. Try asking for specific records like 'show me 1000 products'."
        
        # Check for error information in the DataFrame
        if 'error' in df.columns:
            first_row = df.iloc[0]
            error_msg = first_row['error']
            suggestion = first_row.get('suggestion', '')
            tip = first_row.get('tip', '')
            performance_tip = first_row.get('performance_tip', '')
            
            # Build comprehensive error message
            full_message = f"⚠️ **{error_msg}**"
            
            if suggestion:
                full_message += f"\n\n💡 **Try this:** {suggestion}"
            
            if tip:
                full_message += f"\n\n🎯 **Tip:** {tip}"
                
            if performance_tip:
                full_message += f"\n\n⚡ **Performance:** {performance_tip}"
            
            # Add helpful examples
            full_message += """

🔧 **Example questions that work well:**
- "Show me first 5,000 products"
- "Give me top 10,000 customers" 
- "Display 1,000 recent orders"
- "Show me 500 products from this year"
            """
            
            return None, full_message
        
        # Check for system info (auto-limiting messages)
        system_info = None
        if '_SYSTEM_INFO_' in df.columns:
            system_info = df.iloc[0]['_SYSTEM_INFO_']
            df = df.drop(columns=['_SYSTEM_INFO_'])
            # Display system info as success message
            st.info(f"ℹ️ **System Notice:** {system_info}")
        
        # Check for helpful messages
        if 'message' in df.columns and len(df) == 1:
            first_row = df.iloc[0]
            message = first_row.get('message', '')
            suggestion = first_row.get('suggestion', '')
            
            if 'no data returned' in message.lower():
                msg = f"📭 **{message}**"
                if suggestion:
                    msg += f"\n\n💡 **Suggestion:** {suggestion}"
                return None, msg
        
        # Convert timestamps for Salesforce data
        if data_source == "Salesforce" and df is not None and not df.empty:
            df = convert_salesforce_timestamps(df)
            df = detect_and_convert_timestamps(df)
        
        return df, None
        
    except SnowparkSQLException as e:
        error_str = str(e).lower()
        
        if any(pattern in error_str for pattern in ["timeout", "timed out", "deadline exceeded"]):
            return None, """
⏱️ **Query Timeout - Dataset Too Large**

Your query is processing too many records, causing a timeout.

💡 **Quick Solutions:**
1. **Add a number to your question**: "Show me first 10,000 products"
2. **Be more specific**: "Show me products from last month"
3. **Use filters**: "Show me expensive products"

🎯 **Try these instead:**
- ✅ "Give me top 5,000 product list"
- ✅ "Show me 1,000 recent customers"
- ✅ "Display first 2,000 orders"
- ❌ "Give me all products" (too large)

⚡ **Pro tip**: Start with smaller numbers, then increase as needed!
            """
        else:
            return None, f"""
⚠️ **Database Issue**

There was a problem executing your query.

💡 **Try this:**
1. **Simplify your question**
2. **Add a limit**: "Show me first 1,000 records"  
3. **Be more specific** about what you need

🔧 If the problem persists, contact your administrator.
            """
            
    except Exception as e:
        return None, """
⚠️ **System Error**

An unexpected error occurred.

💡 **Quick fixes:**
1. **Refresh the page** and try again
2. **Ask for fewer records**: "Show me first 1,000 products"
3. **Use simpler questions**

⚡ **Remember**: Large datasets work better with specific limits!
        """

def display_sql_confidence(confidence: dict):
    """Display SQL confidence information."""
    if confidence is None:
        return

    verified_query_used = confidence.get("verified_query_used")
    if verified_query_used is None:
        return

    # Removed UI display for verified query info
    # If needed later, you can restore the st.popover block



def display_sql_query(sql: str, message_index: int, confidence: dict):
    """
    Display SQL query and execute it with properly working pagination.
    FIXED: Page size selection now works correctly.
    """
    current_data_source = st.session_state.selected_yaml
    query_key = f"query_{message_index}_{hash(sql)}"
    
    # Get the default page size from sidebar setting
    default_page_size = st.session_state.get('default_page_size', st.session_state.get('page_size_setting', DEFAULT_PAGE_SIZE))
    
    # Initialize pagination state if not exists - use sidebar default
    if f"page_size_{query_key}" not in st.session_state:
        st.session_state[f"page_size_{query_key}"] = default_page_size
    
    if f"current_page_{query_key}" not in st.session_state:
        st.session_state[f"current_page_{query_key}"] = 1
    
    # Check if query needs modification
    if current_data_source == "Salesforce":
        modified_sql = modify_salesforce_query(sql)
    else:
        modified_sql = sql

    # Display confidence info if available
    display_sql_confidence(confidence)

    # Execute and display results
    with st.expander("📊 Results", expanded=True):
        with st.spinner(f"⚡ Executing via {current_data_source}..."):
            df_full, err_msg = execute_data_procedure(sql, current_data_source)
            
            if df_full is None or not isinstance(df_full, pd.DataFrame):
                if err_msg:
                    st.warning(err_msg)
                else:
                    st.warning("⚠️ Data is not available right now. Please try again later or contact your administrator.")
                return
                
            if df_full.empty:
                st.warning("📭 **No Records Found** - Your query executed successfully but returned no data.")
                return
            
            total_records = len(df_full)
            
            # Determine if pagination is needed (show pagination for more than 25 records)
            needs_pagination = total_records > 25
            
            # Get current pagination state AFTER potential updates from display_pagination_controls
            if needs_pagination:
                # Display pagination controls (this handles all state updates internally)
                display_pagination_controls(query_key, total_records, st.session_state[f"page_size_{query_key}"], st.session_state[f"current_page_{query_key}"])
            
            # NOW get the updated values from session state
            current_page = st.session_state[f"current_page_{query_key}"]
            current_page_size = st.session_state[f"page_size_{query_key}"]
            
            # Show current settings info
            st.info(f"📊 **Dataset** - {total_records:,} records found. Page size: {current_page_size} (from {'sidebar default' if current_page_size == default_page_size else 'custom setting'})")
            
            if needs_pagination:
                # Calculate data slice using the CURRENT page size (which may have been updated)
                start_idx = (current_page - 1) * current_page_size
                end_idx = min(start_idx + current_page_size, total_records)
                df_to_display = df_full.iloc[start_idx:end_idx]
                
            else:
                df_to_display = df_full
                st.success(f"✅ **Complete Dataset** - Showing all {total_records:,} records (no pagination needed)")
            
            # Display results in tabs
            data_tab, chart_tab = st.tabs(["📄 Data", "📈 Chart"])
            
            with data_tab:
                # Export options
                if needs_pagination:
                    export_col1, export_col2 = st.columns([3, 1])
                    with export_col2:
                        csv = df_to_display.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Current Page CSV",
                            data=csv,
                            file_name=f"data_page_{current_page}.csv",
                            mime="text/csv",
                            key=f"csv_download_{query_key}"
                        )
                
                # Display data
                st.dataframe(df_to_display, use_container_width=True, height=400)
                
                # Status information
                if needs_pagination:
                    status_col1, status_col2, status_col3 = st.columns(3)
                    with status_col1:
                        st.metric("📄 Current Page", f"{current_page:,}")
                    with status_col2:
                        st.metric("📊 Records Shown", f"{len(df_to_display):,}")
                    with status_col3:
                        st.metric("🗂️ Total Records", f"{total_records:,}")
                else:
                    st.caption(f"📊 {len(df_to_display)} rows returned")

            with chart_tab:
                # Use current page data for charting
                chart_data = df_to_display
                
                # For very large pages, sample the data
                if len(chart_data) > 1000:
                    chart_data = chart_data.sample(n=1000, random_state=42)
                    st.info("📈 Chart shows a random sample of 1,000 records from current page for performance.")
                
                display_charts_tab(chart_data, message_index)
                
                if needs_pagination:
                    st.caption("📊 Chart shows data from current page only")


def display_charts_tab(df: pd.DataFrame, message_index: int) -> None:
    """
    Display charts tab with real-time aggregation updates.
    FIXED: Charts now update immediately when aggregation method changes.
    """
    if len(df.columns) < 2:
        st.info("📊 At least 2 columns required for charts")
        return
    
    all_cols_set = set(df.columns)
    col1, col2 = st.columns(2)
    
    # Column selectors
    x_col = col1.selectbox(
        "X axis", 
        all_cols_set, 
        key=f"x_col_select_{message_index}"
    )
    y_col = col2.selectbox(
        "Y axis",
        all_cols_set.difference({x_col}),
        key=f"y_col_select_{message_index}",
    )
    
    # Aggregation and chart type selectors
    col3, col4 = st.columns(2)
    aggregation_method = col3.selectbox(
        "Aggregation Method",
        options=["sum", "average", "count", "max", "min"],
        index=0,
        key=f"agg_method_{message_index}",
        help="Choose how to aggregate duplicate x-axis values"
    )
    
    chart_type = col4.selectbox(
        "Select chart type",
        options=["Line Chart 📈", "Bar Chart 📊"],
        key=f"chart_type_{message_index}",
    )
    
    # Create a container for the chart that updates when selections change
    chart_container = st.container()
    
    with chart_container:
        try:
            # Clean the data for charting
            chart_df = df[[x_col, y_col]].dropna().copy()
            
            if len(chart_df) == 0:
                st.warning("No valid data available for charting after cleaning")
                return
            
            # Track if aggregation was applied
            aggregation_applied = False
            original_rows = len(chart_df)
            
            # Check if we need to aggregate (duplicate x values)
            if chart_df[x_col].duplicated().any():
                aggregation_applied = True
                
                if aggregation_method == "sum":
                    if pd.api.types.is_numeric_dtype(chart_df[y_col]):
                        chart_df = chart_df.groupby(x_col)[y_col].sum().reset_index()
                    else:
                        st.warning(f"Cannot sum non-numeric values in {y_col}. Using count instead.")
                        chart_df = chart_df.groupby(x_col)[y_col].count().reset_index()
                        aggregation_method = "count"
                        
                elif aggregation_method == "average":
                    if pd.api.types.is_numeric_dtype(chart_df[y_col]):
                        chart_df = chart_df.groupby(x_col)[y_col].mean().reset_index()
                    else:
                        st.warning(f"Cannot average non-numeric values in {y_col}. Using count instead.")
                        chart_df = chart_df.groupby(x_col)[y_col].count().reset_index()
                        aggregation_method = "count"
                        
                elif aggregation_method == "count":
                    chart_df = chart_df.groupby(x_col)[y_col].count().reset_index()
                    
                elif aggregation_method == "max":
                    if pd.api.types.is_numeric_dtype(chart_df[y_col]):
                        chart_df = chart_df.groupby(x_col)[y_col].max().reset_index()
                    else:
                        # For non-numeric, get the lexicographically maximum value
                        chart_df = chart_df.groupby(x_col)[y_col].max().reset_index()
                        
                elif aggregation_method == "min":
                    if pd.api.types.is_numeric_dtype(chart_df[y_col]):
                        chart_df = chart_df.groupby(x_col)[y_col].min().reset_index()
                    else:
                        # For non-numeric, get the lexicographically minimum value
                        chart_df = chart_df.groupby(x_col)[y_col].min().reset_index()
            
            # Limit chart data points for performance (after aggregation)
            chart_limited = False
            if len(chart_df) > 100:
                chart_df = chart_df.head(100)
                chart_limited = True
            
            # Sort by x-axis for better visualization
            try:
                if pd.api.types.is_numeric_dtype(chart_df[x_col]):
                    chart_df = chart_df.sort_values(x_col)
                elif pd.api.types.is_datetime64_any_dtype(chart_df[x_col]):
                    chart_df = chart_df.sort_values(x_col)
            except:
                pass  # If sorting fails, continue with unsorted data
            
            # Display the chart
            chart_data_for_display = chart_df.set_index(x_col)[y_col]
            
            if chart_type == "Line Chart 📈":
                st.line_chart(chart_data_for_display, height=400)
            elif chart_type == "Bar Chart 📊":
                st.bar_chart(chart_data_for_display, height=400)
            
            # Display informative caption
            caption_parts = []
            
            if aggregation_applied:
                caption_parts.append(f"**{aggregation_method.title()}** of {y_col} grouped by {x_col}")
                caption_parts.append(f"({original_rows} rows → {len(chart_df)} groups)")
            else:
                caption_parts.append(f"{y_col} vs {x_col} (no aggregation needed)")
            
            if chart_limited:
                caption_parts.append("(Limited to first 100 points)")
            
            st.caption("📊 " + " • ".join(caption_parts))
            
            # Show aggregation statistics if applied
            if aggregation_applied:
                with st.expander("📊 Aggregation Details", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Original Rows", f"{original_rows:,}")
                    with col2:
                        st.metric("After Grouping", f"{len(chart_df):,}")
                    with col3:
                        reduction = (1 - len(chart_df)/original_rows) * 100
                        st.metric("Data Reduction", f"{reduction:.1f}%")
                    
                    # Show sample of aggregated data
                    st.subheader("Sample of Aggregated Data:")
                    st.dataframe(chart_df.head(10), use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            st.write("Please try selecting different columns or check your data format.")
            st.write("**Debug Info:**")
            st.write(f"- Selected columns: {x_col}, {y_col}")
            st.write(f"- Data types: {df[x_col].dtype}, {df[y_col].dtype}")
            st.write(f"- Data shape: {df.shape}")



if __name__ == "__main__":
    main()