import pandas as pd
import re
from llama_index.core.readers import SimpleDirectoryReader
from utils.logger import get_logger
import os

logger = get_logger("document_loader_agent")

def parse_emirates_id_details(text: str) -> dict:
    """Extract structured data from Emirates ID text content"""
    extracted_data = {
        "emirates_id": "",
        "name": "",
        "address": "",
        "phone": ""
    }
    
    try:
        # Emirates ID extraction
        id_match = re.search(r"Emirates\s*Id:\s*(\d+)", text, re.IGNORECASE)
        if id_match:
            extracted_data["emirates_id"] = id_match.group(1)
        
        # Name extraction
        name_match = re.search(r"Name:\s*(.+?)\n", text, re.IGNORECASE)
        if name_match:
            extracted_data["name"] = name_match.group(1).strip()
        
        # Address extraction
        address_match = re.search(r"Address:\s*(.+?)\n", text, re.IGNORECASE)
        if address_match:
            extracted_data["address"] = address_match.group(1).strip()
        
        # Phone extraction
        phone_match = re.search(r"Phone:\s*([+\d\s]+)", text, re.IGNORECASE)
        if phone_match:
            extracted_data["phone"] = phone_match.group(1).strip()
        
    except Exception as e:
        logger.error(f"Error parsing Emirates ID details: {str(e)}")
    
    logger.info(f"Extracted Emirates ID details: {extracted_data}")
    return extracted_data

def load_documents_and_extract_fields(emirates_id_file, emirates_id, bank_statement_file):
    # --- PDF Parsing Logic ---
    logger.info(f"Processing Emirates ID file: {'Provided' if emirates_id_file else 'Not provided'}")
    logger.info(f"Processing Bank Statement file: {'Provided' if bank_statement_file else 'Not provided'}")
    
    def parse_pdf(file):
        logger.info("Parsing PDF file using LlamaIndex.")
        try:
            # Create temp directory if not exists
            os.makedirs("temp_uploads", exist_ok=True)
            temp_path = "temp_uploads/temp_uploaded.pdf"
            
            with open(temp_path, "wb") as f:
                f.write(file.read())
                
            docs = SimpleDirectoryReader(input_files=[temp_path]).load_data()
            extracted_text = "\n".join([doc.text for doc in docs])
            logger.info("PDF parsing successful with LlamaIndex.")
            
            # Parse Emirates ID details
            return parse_emirates_id_details(extracted_text)
        except Exception as e:
            logger.error(f"Error parsing PDF with LlamaIndex: {e}")
            return {
                "emirates_id": "",
                "name": "",
                "address": "",
                "phone": ""
            }
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # --- Excel Parsing Logic ---
    def parse_excel(file):
        logger.info("Parsing Excel file, skipping metadata rows.")
        try:
            df = pd.read_excel(file, skiprows=6)  # Skip first 6 rows
            logger.info(f"Excel parsing successful. Shape: {df.shape}")
            logger.info(f"Parsed DataFrame head:\n{df.head().to_string(index=False)}")
            return df
        except Exception as e:
            logger.error(f"Error parsing Excel: {e}")
            return None

    def extract_bank_fields(bank_df):
        extracted_income = 0.0
        extracted_loans = 0.0
        try:
            bank_df.columns = [c.strip() for c in bank_df.columns]
            desc_col = next((c for c in bank_df.columns if 'desc' in c.lower()), None)
            income_col = next((c for c in bank_df.columns if 'income' in c.lower()), None)
            expend_col = next((c for c in bank_df.columns if 'expend' in c.lower()), None)
            date_col = next((c for c in bank_df.columns if 'date' in c.lower()), None)

            if date_col:
                bank_df[date_col] = pd.to_datetime(bank_df[date_col], errors='coerce')

            # Extract monthly average salary
            if desc_col and income_col and date_col:
                salary_rows = bank_df[bank_df[desc_col].str.lower().str.contains('salary', na=False)]
                if not salary_rows.empty:
                    salary_rows = salary_rows.dropna(subset=[income_col, date_col])
                    salary_rows[income_col] = salary_rows[income_col].replace('', 0).fillna(0).astype(float)
                    salary_rows['month'] = salary_rows[date_col].dt.to_period('M')
                    monthly_salary = salary_rows.groupby('month')[income_col].sum()
                    if not monthly_salary.empty:
                        extracted_income = monthly_salary.mean()

            # Extract monthly average EMI
            if desc_col and expend_col and date_col:
                emi_rows = bank_df[bank_df[desc_col].str.lower().str.contains('emi', na=False)]
                if not emi_rows.empty:
                    emi_rows = emi_rows.dropna(subset=[expend_col, date_col])
                    emi_rows[expend_col] = emi_rows[expend_col].replace('', 0).fillna(0).astype(float)
                    emi_rows['month'] = emi_rows[date_col].dt.to_period('M')
                    monthly_emi = emi_rows.groupby('month')[expend_col].sum()
                    if not monthly_emi.empty:
                        extracted_loans = monthly_emi.mean()
        except Exception as e:
            logger.warning(f"Could not extract salary/EMI from bank statement: {e}")
        return extracted_income, extracted_loans

    # Parse Emirates ID
    id_details = parse_pdf(emirates_id_file) if emirates_id_file else {
        "emirates_id": emirates_id,
        "name": "",
        "address": "",
        "phone": ""
    }
    
    # Parse Bank Statement
    bank_df = parse_excel(bank_statement_file) if bank_statement_file else None
    extracted_income, extracted_loans = 0.0, 0.0
    if bank_df is not None:
        extracted_income, extracted_loans = extract_bank_fields(bank_df)
    
    # Log extraction results
    logger.info(f"Document extraction completed - "
                f"Emirates ID: {id_details['emirates_id'][:6] if id_details['emirates_id'] else ''}..., "
                f"Name: {id_details['name'][:10] if id_details['name'] else ''}..., "
                f"Address: {id_details['address'][:10] if id_details['address'] else ''}..., "
                f"Phone: {id_details['phone'][:6] if id_details['phone'] else ''}..., "
                f"Income: {extracted_income}, "
                f"Loans: {extracted_loans}")
    
    return {
        "extracted_emirates_id": id_details["emirates_id"],
        "extracted_name": id_details["name"],
        "extracted_address": id_details["address"],
        "extracted_phone": id_details["phone"],
        "bank_df": bank_df,
        "extracted_income": extracted_income,
        "extracted_loans": extracted_loans
    }