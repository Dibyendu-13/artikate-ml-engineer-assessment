from __future__ import annotations

import random


LABELS = ["billing", "technical_issue", "feature_request", "complaint", "other"]


def generate_dataset(seed: int = 42, per_class: int = 200):
    random.seed(seed)
    templates = {
        "billing": [
            "I was charged twice for {thing}.",
            "Please refund the incorrect invoice for {thing}.",
            "My subscription renewal cost is wrong for {thing}.",
        ],
        "technical_issue": [
            "The {thing} button does nothing when I click it.",
            "I get an error when trying to use {thing}.",
            "The app crashes after I open {thing}.",
        ],
        "feature_request": [
            "Please add {thing} to the dashboard.",
            "I would like a way to export {thing}.",
            "Can you support {thing} in the mobile app?",
        ],
        "complaint": [
            "This is very frustrating because {thing}.",
            "I am unhappy with the service because {thing}.",
            "Your support team was rude about {thing}.",
        ],
        "other": [
            "What are your office hours for {thing}?",
            "I need help updating my account profile.",
            "Can someone call me back about {thing}?",
        ],
    }
    things = ["March", "the export feature", "the premium plan", "last night's order", "my invoice", "the login page"]
    dataset = []
    for label in LABELS:
        for _ in range(per_class):
            tpl = random.choice(templates[label])
            thing = random.choice(things)
            text = tpl.format(thing=thing)
            if label == "billing":
                text += " I was billed incorrectly."
            if label == "technical_issue":
                text += " It fails on Chrome."
            if label == "feature_request":
                text += " That would save us time."
            if label == "complaint":
                text += " This experience is bad."
            if label == "other":
                text += " Thanks."
            dataset.append({"text": text, "label": label})
    random.shuffle(dataset)
    return dataset


def evaluation_set():
    # Hand-written or human-verified examples with intentional overlap so the
    # confusion matrix is meaningful rather than perfectly separable.
    examples = []
    billing = [
        "I was charged twice for the annual subscription.",
        "My invoice includes a fee I do not recognise.",
        "I was billed for a feature I never used.",
        "There is an error in my renewal charge.",
        "The monthly payment appears higher than expected.",
        "Please refund the duplicate charge on my account.",
        "My card was charged after I cancelled.",
        "The invoice amount does not match the plan I bought.",
        "I need help with a billing mistake on my receipt.",
        "Why was I charged for add-ons I did not approve?",
        "My payment failed but the invoice still shows as paid.",
        "I think the subscription fee was applied twice.",
        "The pricing on my bill is not what I agreed to.",
        "Can you check the extra tax on my invoice?",
        "I got a charge after the free trial ended unexpectedly.",
        "The renewal invoice was generated too early.",
        "Please reverse the annual fee from this month.",
        "My account was debited for the wrong plan.",
        "There is a billing dispute on my statement.",
        "I need a corrected invoice for last month.",
    ]
    technical_issue = [
        "The login page keeps throwing an error on Chrome.",
        "The export to CSV button freezes the app.",
        "The dashboard loads slowly and sometimes times out.",
        "The app is broken after the latest update.",
        "I cannot upload files because the spinner never ends.",
        "The mobile app crashes when I open settings.",
        "The password reset link does not work.",
        "I see a blank screen after clicking reports.",
        "The search field returns no results even when items exist.",
        "The app shows a server error when I save changes.",
        "The page layout breaks on Safari.",
        "I get an error when I try to connect my account.",
        "The notification panel refuses to open.",
        "The filter controls are unresponsive in the browser.",
        "The system logs me out without warning.",
        "The checkout flow fails at the final step.",
        "The file download hangs indefinitely.",
        "The app reloads endlessly on mobile.",
        "The password field is not accepting input.",
        "The sync process gets stuck at 99 percent.",
    ]
    feature_request = [
        "Please add a bulk edit feature for invoices.",
        "It would help if you added dark mode to the dashboard.",
        "Please let me export reports to Excel.",
        "Can you support SSO login for enterprise accounts?",
        "I would like a recurring reminder feature.",
        "Please add a filter for archived tickets.",
        "We need an API to download audit logs.",
        "Can you let admins impersonate users?",
        "It would be great to schedule monthly exports.",
        "Please add the ability to merge duplicate profiles.",
        "I want a mobile shortcut for approving requests.",
        "Can you support multi-language labels in the app?",
        "Please add a compare mode for reports.",
        "We need a simpler way to tag conversations.",
        "Can the app send notifications by Slack?",
        "Please expose webhook support for integrations.",
        "I want a dashboard widget for overdue tasks.",
        "Can you add a saved search feature?",
        "Please support CSV import for contacts.",
        "A quick duplicate detection tool would be useful.",
    ]
    complaint = [
        "Your support team was rude and unhelpful.",
        "The service is disappointing and the app keeps failing.",
        "I am unhappy that the site crashed during checkout.",
        "This experience has been very frustrating.",
        "Your product feels unreliable and poorly supported.",
        "I am annoyed that nobody replied to my ticket.",
        "The recent changes made the product harder to use.",
        "This has wasted a lot of my time today.",
        "I am upset that the issue keeps happening.",
        "The service quality is much worse than before.",
        "Your response was dismissive and unhelpful.",
        "I am disappointed by how slow the support has been.",
        "The product keeps failing when I need it most.",
        "This is a terrible experience for our team.",
        "I am frustrated that the bug is still unresolved.",
        "The app keeps breaking and nobody is fixing it.",
        "It is unacceptable that this keeps happening.",
        "We are losing trust because of these problems.",
        "I am unhappy with the poor communication.",
        "The product and support both feel broken.",
    ]
    other = [
        "Can someone call me back about my account?",
        "What are your office hours tomorrow?",
        "Where can I update my phone number?",
        "Please confirm whether my account is active.",
        "Who is the account owner for this workspace?",
        "How do I reset my security questions?",
        "Can I speak with someone about the contract process?",
        "What does the onboarding timeline look like?",
        "Is there a user guide for new admins?",
        "Can you tell me the status of my request?",
        "I need help finding the right contact person.",
        "What is the process for changing my email?",
        "Could someone explain the next steps?",
        "I have a general question about my account.",
        "Who should I ask about access permissions?",
        "Can I get a callback later today?",
        "What time does your team open?",
        "Please send me the documentation link.",
        "How can I verify my account details?",
        "Is there a standard contact for this issue?",
    ]

    for label, bucket in [
        ("billing", billing),
        ("technical_issue", technical_issue),
        ("feature_request", feature_request),
        ("complaint", complaint),
        ("other", other),
    ]:
        examples.extend({"text": text, "label": label} for text in bucket)

    return examples
