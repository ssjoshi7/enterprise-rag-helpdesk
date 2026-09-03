# ── rbac.py — Role-Based Access Control ────────────────────────
# V4.3 Enterprise Security Layer
# Principle: LLM classifies intent. Code decides authorization.

# ── Role definitions ────────────────────────────────────────────
ROLES = ["EMPLOYEE", "IT_SUPPORT", "MANAGER", "ADMIN"]

# ── KB categories accessible per role ──────────────────────────
ROLE_KB_ACCESS = {
    "EMPLOYEE": [
        "PASSWORD", "JIRA", "SALESFORCE", 
        "VPN", "EMAIL", "HARDWARE", 
        "SOFTWARE", "ESCALATION"
    ],
    "IT_SUPPORT": [
        "PASSWORD", "JIRA", "SALESFORCE",
        "VPN", "EMAIL", "HARDWARE",
        "SOFTWARE", "ESCALATION", "IT_INTERNAL"
    ],
    "MANAGER": [
        "PASSWORD", "JIRA", "SALESFORCE",
        "VPN", "EMAIL", "HARDWARE",
        "SOFTWARE", "ESCALATION", "MANAGER_DOCS"
    ],
    "ADMIN": ["ALL"]  # Admin sees everything
}

# ── Action permissions per role ─────────────────────────────────
ROLE_ACTIONS = {
    "EMPLOYEE": {
        "create_ticket": True,
        "view_all_tickets": False,
        "view_team_tickets": False,
        "access_restricted_kb": False,
        "manage_users": False
    },
    "IT_SUPPORT": {
        "create_ticket": True,
        "view_all_tickets": True,
        "view_team_tickets": True,
        "access_restricted_kb": False,
        "manage_users": False
    },
    "MANAGER": {
        "create_ticket": True,
        "view_all_tickets": False,
        "view_team_tickets": True,
        "access_restricted_kb": False,
        "manage_users": False
    },
    "ADMIN": {
        "create_ticket": True,
        "view_all_tickets": True,
        "view_team_tickets": True,
        "access_restricted_kb": True,
        "manage_users": True
    }
}

# ── Role lookup — deterministic, not LLM-based ─────────────────
def get_user_role(email, secrets):
    try:
        roles_config = secrets.get("roles", {})
        # Check each role mapping
        if email == roles_config.get("admin_email"):
            return "ADMIN"
        elif email == roles_config.get("it_support_email"):
            return "IT_SUPPORT"
        elif email == roles_config.get("manager_email"):
            return "MANAGER"
        elif email == roles_config.get("employee_email"):
            return "EMPLOYEE"
        else:
            return "EMPLOYEE"  # safe default
    except Exception as e:
        print(f"   ⚠️ Role lookup failed: {e} — defaulting to EMPLOYEE")
        return "EMPLOYEE"

# ── KB access check ─────────────────────────────────────────────
def can_access_category(role, category):
    """
    Check if role can access a KB category.
    Deterministic authorization — not LLM-based.
    """
    allowed = ROLE_KB_ACCESS.get(role, ROLE_KB_ACCESS["EMPLOYEE"])
    if "ALL" in allowed:
        return True
    return category in allowed

# ── Action authorization check ──────────────────────────────────
def can_perform_action(role, action):
    """
    Check if role can perform an action.
    Deterministic authorization — not LLM-based.
    """
    permissions = ROLE_ACTIONS.get(role, ROLE_ACTIONS["EMPLOYEE"])
    return permissions.get(action, False)

# ── Get allowed categories for role ────────────────────────────
def get_allowed_categories(role):
    """
    Return list of KB categories this role can access.
    Used to filter retrieval.
    """
    allowed = ROLE_KB_ACCESS.get(role, ROLE_KB_ACCESS["EMPLOYEE"])
    if "ALL" in allowed:
        return None  # None means no filter — access everything
    return allowed

# ── Role display info ───────────────────────────────────────────
ROLE_DISPLAY = {
    "EMPLOYEE": {"label": "Employee", "color": "blue", "icon": "👤"},
    "IT_SUPPORT": {"label": "IT Support", "color": "green", "icon": "🛠️"},
    "MANAGER": {"label": "Manager", "color": "orange", "icon": "👔"},
    "ADMIN": {"label": "Administrator", "color": "red", "icon": "🔐"}
}

def get_role_display(role):
    return ROLE_DISPLAY.get(role, ROLE_DISPLAY["EMPLOYEE"])