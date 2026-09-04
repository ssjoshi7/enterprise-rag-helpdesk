# ── jsm_ticketing.py — Jira Service Management Integration ──────
# V4.5 Track 2 — Replace Airtable with enterprise JSM ticketing

import os
import requests
import base64

# ── Load JSM credentials ────────────────────────────────────────
def get_jsm_credentials():
    """Load JSM credentials from env or Streamlit secrets."""
    # Try env first
    creds = {
        "email": os.getenv("JSM_EMAIL"),
        "token": os.getenv("JSM_API_TOKEN"),
        "site": os.getenv("JSM_SITE"),
        "project_key": os.getenv("JSM_PROJECT_KEY")
    }
    
    # If any missing — try Streamlit secrets
    if not all(creds.values()):
        try:
            import streamlit as st
            creds = {
                "email": st.secrets.get("JSM_EMAIL"),
                "token": st.secrets.get("JSM_API_TOKEN"),
                "site": st.secrets.get("JSM_SITE"),
                "project_key": st.secrets.get("JSM_PROJECT_KEY")
            }
        except Exception:
            pass
    
    return creds

# ── Get auth header ─────────────────────────────────────────────
def get_auth_header(email, token):
    """Generate Basic Auth header for Atlassian API."""
    credentials = f"{email}:{token}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

# ── Get service desk ID ─────────────────────────────────────────
def get_service_desk_id(creds):
    """Get the service desk ID for the project."""
    url = f"https://{creds['site']}/rest/servicedeskapi/servicedesk"
    headers = get_auth_header(creds["email"], creds["token"])
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            desks = response.json().get("values", [])
            for desk in desks:
                if desk.get("projectKey") == creds["project_key"]:
                    return desk.get("id")
        print(f"   ⚠️ Could not find service desk: {response.status_code}")
        return None
    except Exception as e:
        print(f"   ❌ Service desk lookup failed: {e}")
        return None

# ── Get request type ID ─────────────────────────────────────────
def get_request_type_id(creds, service_desk_id):
    """Get the first available request type ID."""
    url = f"https://{creds['site']}/rest/servicedeskapi/servicedesk/{service_desk_id}/requesttype"
    headers = get_auth_header(creds["email"], creds["token"])
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            types = response.json().get("values", [])
            if types:
                return types[0].get("id")
        print(f"   ⚠️ Could not find request types: {response.status_code}")
        return None
    except Exception as e:
        print(f"   ❌ Request type lookup failed: {e}")
        return None

# ── Create JSM ticket ───────────────────────────────────────────
def create_jsm_ticket(actual_issue, user_email=None, urgency="Medium"):
    """
    Create a real Jira Service Management ticket.
    Returns ticket ID and URL if successful.
    """
    print(f"   🎫 Creating JSM ticket for: '{actual_issue}'")
    
    creds = get_jsm_credentials()
    
    if not all([creds["email"], creds["token"], creds["site"], creds["project_key"]]):
        print(f"   ❌ JSM credentials missing")
        return None, None
    
    headers = get_auth_header(creds["email"], creds["token"])
    
    # Step 1 — Get service desk ID
    service_desk_id = get_service_desk_id(creds)
    if not service_desk_id:
        return None, None
    
    # Step 2 — Get request type ID
    request_type_id = get_request_type_id(creds, service_desk_id)
    if not request_type_id:
        return None, None
    
    # Step 3 — Create ticket
    url = f"https://{creds['site']}/rest/servicedeskapi/request"
    
    payload = {
        "serviceDeskId": str(service_desk_id),
        "requestTypeId": str(request_type_id),
        "requestFieldValues": {
            "summary": actual_issue,
            "description": f"Issue reported by: {user_email or 'Unknown'}\n\nIssue: {actual_issue}\nPriority: {urgency}\n\nCreated via IT Helpdesk RAG Agent"
        }
    }
    
    MAX_RETRIES = 2
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"   🔄 JSM API attempt {attempt}/{MAX_RETRIES}")
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                ticket_id = data.get("issueKey", data.get("issueId", "Unknown"))
                ticket_url = f"https://{creds['site']}/servicedesk/customer/portal/{service_desk_id}/{ticket_id}"
                print(f"   ✅ JSM ticket created: {ticket_id}")
                return ticket_id, ticket_url
                
            elif response.status_code == 401:
                print(f"   ❌ JSM authentication failed")
                return None, None
                
            else:
                print(f"   ⚠️ JSM returned {response.status_code}: {response.text[:200]}")
                if attempt == MAX_RETRIES:
                    return None, None
                    
        except requests.exceptions.Timeout:
            print(f"   ⚠️ JSM timeout on attempt {attempt}")
            if attempt == MAX_RETRIES:
                return None, None
        except Exception as e:
            print(f"   ⚠️ JSM error on attempt {attempt}: {e}")
            if attempt == MAX_RETRIES:
                return None, None
    
    return None, None