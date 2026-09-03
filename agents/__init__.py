from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.acmpo import ACMPOAgent
from agents.acmpo_original import ACMPOOriginalAgent

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    acmpo=ACMPOAgent,
    acmpo_original=ACMPOOriginalAgent,
)
