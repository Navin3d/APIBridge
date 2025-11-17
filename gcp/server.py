import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.adk.cli.fast_api import get_fast_api_app
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4
from datetime import datetime

# Load environment variables
load_dotenv()

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
app_args = {"agents_dir": AGENT_DIR, "web": True}

# Create FastAPI app with ADK integration
app: FastAPI = get_fast_api_app(**app_args)

# Update app metadata
app.title = "Production ADK Agent - Lab 3"
app.description = "Gemma agent with GPU-accelerated backend"
app.version = "1.0.0"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "production-adk-agent"}

@app.get("/")
def root():
    return {
        "service": "Production ADK Agent - Lab 3",
        "description": "GPU-accelerated Gemma agent",
        "docs": "/docs",
        "health": "/health"
    }

class PaymentInitiationRequest(BaseModel):
    """
    Request body for initiating a payment.
    Fields:
      debtor_account (str): The account number of the sender.
      creditor_account (str): The account number of the recipient.
      amount (float): Amount to send.
      currency (str): Currency code (e.g. 'INR', 'USD').
      reference (Optional[str]): Payment reference (optional).
    """
    debtor_account: str
    creditor_account: str
    amount: float
    currency: str
    reference: Optional[str]

class PaymentInitiationResponse(BaseModel):
    """
    Response body for payment initiation.
    Fields:
      payment_id (str): Unique identifier for the payment.
      status (str): Status of the payment (e.g. 'PENDING').
      created_at (str): ISO timestamp of creation.
    """
    payment_id: str
    status: str
    created_at: str

class PaymentStatusResponse(BaseModel):
    """
    Response for payment status query.
    Fields:
      payment_id (str): Unique identifier for the payment.
      status (str): Current status (e.g. 'PENDING', 'COMPLETED').
      processed_at (Optional[str]): ISO timestamp when processed, or None if pending.
      amount (float): Payment amount.
      currency (str): Currency code.
      reference (Optional[str]): Payment reference (optional).
    """
    payment_id: str
    status: str
    processed_at: Optional[str]
    amount: float
    currency: str
    reference: Optional[str]

class PayoutRequest(BaseModel):
    """
    Request body for sending a payout.
    Fields:
      payee_name (str): Name of the payout recipient.
      payee_bank_details (str): Bank details (e.g., account or IFSC).
      amount (float): Amount to send.
      currency (str): Currency code.
      reference (Optional[str]): Payment reference (optional).
    """
    payee_name: str
    payee_bank_details: str
    amount: float
    currency: str
    reference: Optional[str]

class PayoutResponse(BaseModel):
    """
    Response for payout initiation.
    Fields:
      payout_id (str): Unique identifier for this payout.
      status (str): Status (e.g., 'SENT').
      sent_at (str): UTC ISO timestamp when sent.
    """
    payout_id: str
    status: str
    sent_at: str

# In-memory store for demonstration purposes only
payments_store = {}
payouts_store = {}

@app.post("/payments/initiate", response_model=PaymentInitiationResponse)
def initiate_payment(request: PaymentInitiationRequest):
    """
    Initiate a single payment by providing sender, recipient, amount and currency details.

    Args:
        request (PaymentInitiationRequest): Payment initiation payload.

    Returns:
        PaymentInitiationResponse: Contains new payment ID, status, and creation time.
    """
    payment_id = str(uuid4())
    payments_store[payment_id] = {
        "status": "PENDING",
        "amount": request.amount,
        "currency": request.currency,
        "reference": request.reference,
        "created_at": datetime.utcnow().isoformat()
    }
    return PaymentInitiationResponse(
        payment_id=payment_id,
        status="PENDING",
        created_at=payments_store[payment_id]["created_at"]
    )

@app.get("/payments/{payment_id}/status", response_model=PaymentStatusResponse)
def get_payment_status(payment_id: str):
    """
    Check the current status and details of a specific payment using its payment_id.

    Args:
        payment_id (str): The unique payment identifier returned at creation.

    Returns:
        PaymentStatusResponse: Contains status, processed timestamp, amount, currency and reference.
    Raises:
        HTTPException: If the payment_id does not exist.
    """
    payment = payments_store.get(payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")
    return PaymentStatusResponse(
        payment_id=payment_id,
        status=payment["status"],
        processed_at=(datetime.utcnow().isoformat() if payment["status"] == "COMPLETED" else None),
        amount=payment["amount"],
        currency=payment["currency"],
        reference=payment["reference"],
    )

@app.post("/payouts/send", response_model=PayoutResponse)
def send_payout(request: PayoutRequest):
    """
    Initiate a payout to a recipient’s account.

    Args:
        request (PayoutRequest): Payout details including payee name, bank details, amount, and currency.

    Returns:
        PayoutResponse: Contains new payout ID, status, and sent timestamp.
    """
    payout_id = str(uuid4())
    payouts_store[payout_id] = {
        "status": "SENT",
        "amount": request.amount,
        "currency": request.currency,
        "sent_at": datetime.utcnow().isoformat()
    }
    return PayoutResponse(
        payout_id=payout_id,
        status="SENT",
        sent_at=payouts_store[payout_id]["sent_at"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
