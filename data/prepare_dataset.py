"""
Dataset Preparation & Loader for Banking77 (10 Intent Subset)
Curates realistic, high-fidelity customer support queries across 10 banking intents.
Provides stratified train, validation, and test splits.
"""

import json
import os
from typing import Dict, List, Tuple

DATASET_RECORDS = {
    "card_arrival": [
        # Train (12)
        "When will my new card arrive in the mail?",
        "I ordered a replacement card 5 days ago, how long does delivery usually take?",
        "Is there a tracking number available for my physical card shipment?",
        "My card has not arrived yet and it has been two weeks.",
        "Can you tell me the expected delivery date for my debit card?",
        "Where is my card? It should have arrived by Monday.",
        "Will I receive a postal notification when my card is dispatched?",
        "How many business days do I need to wait for my new card to get here?",
        "I need my card before traveling on Friday, has it shipped out?",
        "What postal service is delivering my debit card?",
        "My tracking says card delivered but I haven't received anything.",
        "Can I expedite the shipping of my replacement card?",
        # Val (4)
        "Has my new payment card been dispatched yet?",
        "Estimated arrival time for my standard delivery debit card?",
        "Still waiting on my physical card, is there a delay?",
        "I was told 3-5 days for card arrival but it is day 7.",
        # Test (8)
        "When can I expect my debit card to reach my home address?",
        "I haven't received my card yet, can you check shipping status?",
        "How long does standard delivery take for debit cards in the UK?",
        "My card should have arrived yesterday, what should I do?",
        "Is my card on its way? It's been over a week since ordering.",
        "Do you provide a tracking link for card deliveries?",
        "Can someone verify if my replacement card has been mailed?",
        "My card is overdue, did it get lost in transit?"
    ],
    "card_linking": [
        # Train (12)
        "How do I link an external debit card to my app account?",
        "Can I connect my Chase bank card to this account?",
        "Is it possible to add a secondary payment card to my wallet?",
        "I am having trouble linking my Mastercard, it keeps throwing an error.",
        "Steps to attach another debit card for funding my account?",
        "Why won't my Visa card connect to the app?",
        "Can I link a card that is not in my legal name?",
        "How many external cards can I link to my account at once?",
        "Where in settings can I manage my linked cards?",
        "I want to connect my credit card to top up automatically.",
        "My bank blocked the verification charge when linking my card.",
        "Can I link a prepaid card from another provider?",
        # Val (4)
        "How do I add my existing debit card to the profile?",
        "It says card linking failed, please try again later.",
        "I want to connect a new payment card for deposits.",
        "Can I link an American Express card to this app?",
        # Test (8)
        "What is the procedure for linking a new card to my account?",
        "Why is the app refusing to connect my external debit card?",
        "I would like to link my corporate credit card to my balance.",
        "Is there a limit on how many cards I can attach to my profile?",
        "How do I verify the small deposit when linking a new card?",
        "Can I connect my overseas bank card to this system?",
        "The app crashed while I was trying to link my card.",
        "How do I remove an old linked card and add a fresh one?"
    ],
    "exchange_rate": [
        # Train (12)
        "What foreign exchange rate do you use for converting EUR to USD?",
        "Do you charge an extra markup on currency exchanges during weekends?",
        "Where can I see live exchange rates in the application?",
        "How is the exchange rate calculated when I pay abroad?",
        "Is the exchange rate fixed or floating during card transactions?",
        "Why was the conversion rate different from what Google showed?",
        "Are there hidden fees baked into the currency exchange rate?",
        "What is the weekend exchange rate policy for British Pounds?",
        "Can I lock in a specific foreign currency exchange rate in advance?",
        "Do you use the interbank exchange rate for currency conversions?",
        "How much will it cost to convert 500 Euros into Japanese Yen?",
        "Is there a limit on how much currency I can convert at the real exchange rate?",
        # Val (4)
        "What is your current currency exchange fee structure?",
        "Will I get the interbank rate if I exchange money today?",
        "Why did the exchange rate change right before I completed the swap?",
        "Does your foreign currency exchange include a markup percentage?",
        # Test (8)
        "What exchange rate applies when I spend money in Europe?",
        "Can you explain how currency conversion rates are determined?",
        "Is there a weekend fee added on top of the foreign exchange rate?",
        "Where can I check the live conversion rate between GBP and USD?",
        "Why is the exchange rate higher today than yesterday?",
        "Do you charge commission on top of the interbank exchange rate?",
        "How do I check the exchange rate for Polish Zloty before sending?",
        "What FX rate is applied when withdrawing local currency from foreign ATMs?"
    ],
    "extra_charge_on_statement": [
        # Train (12)
        "There is an unexpected charge of $14.99 on my bank statement.",
        "I see a mystery fee on my account that I never authorized.",
        "Why was I billed an extra $5 on my monthly statement?",
        "There is an unknown recurring charge listed on my statement summary.",
        "I noticed an extra transaction on my statement from a vendor I don't know.",
        "Why did my account balance drop by $20 without any receipt?",
        "I was charged twice for the same subscription on my statement.",
        "Who authorized this unfamiliar fee on my statement?",
        "Can you investigate an unauthorized charge showing on my latest statement?",
        "There is a $2.50 maintenance fee on my statement that wasn't there before.",
        "Why is there a duplicate transaction on my statement for grocery shopping?",
        "I see an extra charge that appeared overnight on my account statement.",
        # Val (4)
        "I have an unexplained charge on my statement from yesterday.",
        "Why does my statement show an additional unexpected fee?",
        "There's a charge on my bank statement that I did not make.",
        "Can you explain this extra deduction showing on my statement?",
        # Test (8)
        "There is an unrecognized charge of $32 on my statement, what is it for?",
        "I noticed an extra fee on my recent account statement that looks suspicious.",
        "Why do I have an extra line item charge on my statement this month?",
        "Can you help me dispute an unexpected charge on my statement?",
        "My statement shows two charges from the same coffee shop at the exact same minute.",
        "What is this mystery $10 debit on my monthly statement?",
        "There is an extra charge on my card statement from an unknown company.",
        "I don't recognize the latest charge on my statement summary."
    ],
    "lost_or_stolen_card": [
        # Train (12)
        "I lost my wallet yesterday and need to freeze my card immediately.",
        "My debit card was stolen out of my backpack on the train.",
        "Help! Someone took my card and is making fraudulent transactions.",
        "How do I permanently cancel a stolen card and request a replacement?",
        "I think I left my card in an ATM, can you block it right away?",
        "My card is missing, how do I temporarily freeze it from the app?",
        "Someone pickpocketed my card, please lock my account now.",
        "I suspect my card details were copied by a skimmer, please disable it.",
        "I can't find my physical card anywhere, should I cancel it?",
        "Emergency: my credit card was stolen in Barcelona, please deactivate it.",
        "Can you lock my stolen card so no one can tap to pay?",
        "I lost my card at a concert, how quickly can you issue a new one?",
        # Val (4)
        "My card was stolen this morning, please block it immediately.",
        "I dropped my card on the street and can't find it, what should I do?",
        "Someone stole my purse containing my payment card.",
        "Please freeze my card right now because it has gone missing.",
        # Test (8)
        "I lost my physical card, can you deactivate it immediately?",
        "My card was stolen from my gym locker, please block all charges.",
        "Someone is using my stolen card, please cancel it urgently!",
        "How do I report my debit card as lost and get a new one?",
        "I misplaced my card in a taxi, can I temporarily freeze it?",
        "My card was taken along with my wallet, please freeze it.",
        "I think someone stole my card information, can you void the card?",
        "What should I do first if my card has been stolen while overseas?"
    ],
    "top_up_reverted": [
        # Train (12)
        "My top up was reverted back to my originating bank account.",
        "Why was my deposit of $200 cancelled and sent back?",
        "I added funds yesterday but the top-up transaction was reverted today.",
        "My top-up failed and the money went back into my external account.",
        "The application says my recent top-up has been reverted.",
        "Why did you reject my balance deposit and reverse the transaction?",
        "My money showed in my balance for an hour and then was reverted.",
        "What caused my card top-up to bounce back to my main bank?",
        "I topped up $100 but received a notification that it was reversed.",
        "How long until reverted top-up money is available back in my bank?",
        "Can you explain why my recurring top up was automatically reverted?",
        "The system cancelled my top-up payment and refunded it.",
        # Val (4)
        "Why has my recent top up payment been reverted?",
        "My balance deposit was cancelled and the funds were reverted.",
        "The top up I did this morning was reverted without explanation.",
        "Why did my money top-up bounce back to my card?",
        # Test (8)
        "Why was my $150 top up transaction reverted by your system?",
        "I tried topping up my account but the funds were automatically sent back.",
        "My top-up says reverted, when will my external bank show the refund?",
        "What are the typical reasons for a top-up being reverted?",
        "The app reverted my deposit after initially approving it.",
        "Why was my debit card top-up cancelled and returned to sender?",
        "My top up bounced back and my balance is back to zero.",
        "Can you un-revert a top up or do I have to initiate a new deposit?"
    ],
    "transfer_fee_charged": [
        # Train (12)
        "Why was I charged a $12 fee for sending money to my friend?",
        "Do you charge a transfer fee for domestic bank transfers?",
        "I didn't expect a commission fee on my international wire transfer.",
        "How much is the transfer fee for sending money to an overseas account?",
        "Can you break down the fees charged on my recent SEPA transfer?",
        "Why was there a transaction fee deducted when I transferred money?",
        "Are domestic peer-to-peer transfers completely free of charge?",
        "I was charged an extra fee when sending money to another user.",
        "What is the fee policy for SWIFT wire transfers?",
        "Why did the recipient receive less money due to transfer fees?",
        "Can I get a refund for the transfer fee charged by mistake?",
        "Is there a fee for transferring money via local bank transfer?",
        # Val (4)
        "Why did you charge me a transfer fee on this transaction?",
        "What is the cost or fee for sending a bank transfer abroad?",
        "I was surprised to see a transfer fee deducted from my payment.",
        "How much are transfer fees for sending money between accounts?",
        # Test (8)
        "Why was a fee applied to my money transfer yesterday?",
        "How much is the fee for an international bank transfer?",
        "I thought transfers were free, why was a transfer charge deducted?",
        "Can you explain the transfer fee shown on my payment confirmation?",
        "Why was $15 taken off as a transfer fee for my wire transfer?",
        "Does sending funds in another currency incur a transfer commission?",
        "Are there any fees when I transfer funds to another member?",
        "Why did my beneficiary get less money after transfer fees were taken?"
    ],
    "verify_identity": [
        # Train (12)
        "How do I complete my identity verification in the mobile app?",
        "My passport photo keeps getting rejected during verification.",
        "Why is my KYC identity check still showing as pending after 3 days?",
        "What documents are accepted to verify my legal identity?",
        "Can I upload a national identity card instead of a passport for verification?",
        "I am having trouble with the selfie verification camera step.",
        "Why do I need to verify my identity before withdrawing funds?",
        "How long does the identity verification review process take?",
        "The app says my proof of address is expired, how do I re-verify?",
        "Can you manually approve my identity documents?",
        "Where in account settings can I re-submit my identity verification photos?",
        "My driver's license was rejected during identity verification.",
        # Val (4)
        "What is the status of my identity verification check?",
        "How can I submit my documents to verify my identity?",
        "Why has my identity verification not been approved yet?",
        "I am stuck on the ID verification screen in the app.",
        # Test (8)
        "What documents do I need to submit to verify my identity?",
        "Why is my identity verification still pending after 48 hours?",
        "My selfie verification failed multiple times, can someone help?",
        "Can I verify my identity using a residence permit?",
        "How do I know if my identity verification was approved successfully?",
        "Why was my passport rejected during the identity verification step?",
        "Is it mandatory to complete identity verification to send money?",
        "The camera won't focus on my document during the identity check."
    ],
    "visa_or_mastercard": [
        # Train (12)
        "Is the debit card you provide a Visa or a Mastercard?",
        "Can I choose to have a Mastercard instead of a Visa card?",
        "What payment brand network does your card use?",
        "Do you offer both Visa and Mastercard options for customer cards?",
        "Are there any differences in merchant acceptance between your Visa and Mastercard?",
        "Why did I receive a Mastercard when my friend got a Visa?",
        "Can I request a Visa card specifically for traveling to Asia?",
        "Is your virtual card powered by Mastercard or Visa?",
        "Does the card brand affect foreign exchange fees?",
        "I only want a Visa debit card, can I request one during signup?",
        "Which card scheme do your commercial cards run on?",
        "Can I hold both a Visa and a Mastercard on the same account?",
        # Val (4)
        "Do you issue Visa cards or Mastercard cards?",
        "Can I select between Visa and Mastercard when ordering my card?",
        "Is my new physical card going to be a Visa or Mastercard?",
        "What payment network brand is printed on the physical card?",
        # Test (8)
        "Is my card a Visa card or a Mastercard?",
        "Can I switch my existing card from Visa to Mastercard?",
        "Why does my digital card show Mastercard but my friend has Visa?",
        "Do you have an option to order a Visa debit card?",
        "What are the benefits of Mastercard compared to Visa on this account?",
        "Are both Visa and Mastercard accepted globally everywhere?",
        "Which card network is used for your premium cards?",
        "Can I get a secondary card issued by Visa if my main is Mastercard?"
    ],
    "wrong_amount_additional_charge": [
        # Train (12)
        "The restaurant charged me $45 but my receipt clearly says $35.",
        "I was charged the wrong amount for my grocery store purchase.",
        "The store overcharged me by $10 on my card payment.",
        "Why does the charged amount not match what the cashier told me?",
        "My gas station pre-authorization charged a higher amount than the pump.",
        "The hotel billed me for $180 instead of the agreed $120 booking fee.",
        "I was overcharged on my card for taxi fare yesterday.",
        "The merchant entered the wrong total amount on the card terminal.",
        "Can I dispute a transaction because the merchant charged the wrong amount?",
        "My invoice says $80 but $95 was debited from my account.",
        "I was charged an additional unexpected tip that I never authorized.",
        "The final bill amount is higher than the price shown on the checkout screen.",
        # Val (4)
        "The merchant charged me more money than the agreed total.",
        "Why was I billed an incorrect higher amount for this purchase?",
        "I have a receipt showing $25 but my transaction history says $32.",
        "The store charged me the wrong amount on my debit card.",
        # Test (8)
        "The merchant debited the wrong total amount from my account.",
        "I was charged $60 instead of $40 at the restaurant last night.",
        "My receipt shows $15.50 but my card was charged $25.50.",
        "How do I dispute a transaction where the merchant entered the wrong amount?",
        "The shopkeeper charged me twice the price by mistake.",
        "Why was the charge higher than the bill I signed for?",
        "I was overcharged by $15 on my recent card payment, can I get it refunded?",
        "The retailer charged an incorrect extra sum on top of my order price."
    ]
}


def build_and_save_dataset():
    """Builds and writes the complete structured JSON dataset."""
    train_records = []
    val_records = []
    test_records = []

    intents = sorted(list(DATASET_RECORDS.keys()))

    for intent in intents:
        examples = DATASET_RECORDS[intent]
        train_examples = examples[:12]
        val_examples = examples[12:16]
        test_examples = examples[16:24]

        for text in train_examples:
            train_records.append({"text": text, "intent": intent, "split": "train"})
        for text in val_examples:
            val_records.append({"text": text, "intent": intent, "split": "val"})
        for text in test_examples:
            test_records.append({"text": text, "intent": intent, "split": "test"})

    dataset = {
        "metadata": {
            "name": "Banking77-10Intents",
            "task": "Customer Support Intent Classification",
            "num_intents": len(intents),
            "intents": intents,
            "counts": {
                "train": len(train_records),
                "val": len(val_records),
                "test": len(test_records),
                "total": len(train_records) + len(val_records) + len(test_records)
            }
        },
        "train": train_records,
        "val": val_records,
        "test": test_records
    }

    output_path = os.path.join(
        os.path.dirname(__file__), "banking77_subsets", "banking77_10intents.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved successfully to {output_path}")
    print(f"Stats: Train={len(train_records)} | Val={len(val_records)} | Test={len(test_records)} | Total={dataset['metadata']['counts']['total']}")
    return dataset


if __name__ == "__main__":
    build_and_save_dataset()
