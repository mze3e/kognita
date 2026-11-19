"""Generate PWM module 13 (business scope: costs and charges, product governance, client circumstances,
operations and lifecycle, performance provenance): ontology, shapes, example data and negative fixtures.
Reuses the helpers in gen_module12.py. Edit this script, not the generated files."""
import json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_module12 as g
ROOT = g.ROOT
g.NEWP = "0.10: NEW specialized record or property; no exact standard term identified"
g.LOC.update({
 "B1":"MiFID II Art. 24(4)(c) costs and charges information; Delegated Regulation 2017/565 Art. 50",
 "B2":"MiFID II Art. 24(9) inducements; Delegated Regulation 2017/565 Art. 11",
 "B3":"MiFID II Art. 16(3) and Art. 24(2) product governance, target market",
 "B4":"FCA FG21/1 four drivers of vulnerability: health, life events, resilience, capability",
 "B5":"MiFID II Art. 25(2) suitability, periodic assessment under Delegated Regulation 2017/565 Art. 54(12)",
 "B6":"MiFID II Art. 27 best execution; Delegated Regulation 2017/565 Art. 64",
 "B7":"OECD Common Reporting Standard; US FATCA reporting",
 "B8":"PWM scope: complaint handling, corporate actions, income, lending and performance provenance",
})
# refs per key: B1,B2,B3,B5,B6 -> EU-MIFID2 ; B4 -> UK-FCA-FG21-1 ; B7 -> OECD-CRS ; B8 -> KOG-ROADMAP not applicable -> PWM-PRACTICE only
REF = {"B1":"EU-MIFID2","B2":"EU-MIFID2","B3":"EU-MIFID2","B4":"UK-FCA-FG21-1","B5":"EU-MIFID2","B6":"EU-MIFID2","B7":"OECD-CRS"}
def cit(*keys):
    parts=[]
    for k in keys:
        parts.append(f'[ pwm:cites pwmref:{REF.get(k,"PWM-PRACTICE")} ; pwm:locator "{g.LOC[k]}" ]')
    parts.append(f'[ pwm:cites pwmref:PWM-PRACTICE ; pwm:locator "{g.NEWP}" ]')
    return ", ".join(parts)
C = [
("CostDisclosure","cost disclosure","Disclosure to a client of the costs and charges of an investment service or product, ex ante before the service and ex post after the period, with the total and its items.","prov:Entity","Medium","C2",["B1"]),
("CostItem","cost item","One item in a cost disclosure: category, amount and currency, optionally the fee it comes from.","prov:Entity","Medium","C2",["B1"]),
("InducementDisclosure","inducement disclosure","Disclosure of a third-party payment, with whether it is designed to enhance the quality of the service to the client.","prov:Entity","Medium","C2",["B2"]),
("ProductApproval","product approval","Approval of a product for distribution to a target market, with the distribution strategy that applies.","prov:Entity","Medium","C1",["B3"]),
("TargetMarketReview","target market review","Periodic review of a product's target market, with the outcome and the next review date.","prov:Entity","Medium","C1",["B3"]),
("ComplaintOutcome","complaint outcome","Resolution of a complaint: result, date, any redress and the root cause.","prov:Entity","Medium","C2",["B8"]),
("SuitabilityReview","suitability review","Periodic review of a client's suitability assessment, with its result and the next review date.","prov:Entity","Medium","C2",["B5"]),
("Order","order","Instruction to buy or sell an instrument in an account, under a mandate or on the client's instruction.","prov:Entity","Medium","C2",["B6"]),
("Execution","execution","Fill of an order: quantity, price, time and venue.","prov:Entity","Medium","C2",["B6"]),
("Allocation","allocation","Assignment of an executed quantity to an account.","prov:Entity","Medium","C2",["B6"]),
("BestExecutionRecord","best execution record","Evidence of how an execution met the best execution policy: the factors considered and the rationale for the venue.","prov:Entity","Medium","C1",["B6"]),
("CorporateAction","corporate action","Event that changes an instrument or pays its holders: dividend, coupon, split, merger, rights issue or redemption.","prov:Entity","Medium","C1",["B8"]),
("IncomeReceipt","income receipt","Income credited to an account, optionally from a corporate action.","prov:Entity","Medium","C2",["B8"]),
("TaxReportingRecord","tax reporting record","Record that an account was reported under CRS or FATCA for a year, to a jurisdiction, with the balance reported.","prov:Entity","Medium","C3",["B7"]),
("LendingFacility","lending facility","Credit facility to a client: Lombard, mortgage, structured loan or other, with limit and maximum loan-to-value.","prov:Entity","Medium","C2",["B8"]),
("PerformanceFigure","performance figure","Performance of a portfolio over a period, as a figure from a registered calculation service. PWM records its provenance and never computes it.","prov:Entity","Medium","C2",["B8"]),
]
P = [
("disclosureTiming","c","disclosure timing","Whether the disclosure is ex ante or ex post.","pwm:CostDisclosure","skos:Concept",["B1"],"DisclosureTimingScheme"),
("disclosedToClient","o","disclosed to client","Client who received the disclosure.",None,"pwm:Client",["B1","B2"]),
("disclosedAt","d","disclosed at","When the disclosure was made.",None,"xsd:dateTime",["B1","B2"]),
("coversMandate","o","covers mandate","Mandate whose service the disclosure covers.","pwm:CostDisclosure","pwm:Mandate",["B1"]),
("totalCostAmount","d","total cost amount","Total cost over the disclosure period.","pwm:CostDisclosure","xsd:decimal",["B1"]),
("totalCostPercent","d","total cost percent","Total cost as a percentage of the invested amount.","pwm:CostDisclosure","xsd:decimal",["B1"]),
("hasCostItem","o","has cost item","Item that makes up the disclosed cost.","pwm:CostDisclosure","pwm:CostItem",["B1"]),
("includesInducements","d","includes inducements","Whether the disclosed cost includes third-party payments.","pwm:CostDisclosure","xsd:boolean",["B1","B2"]),
("periodStart","d","period start","Start of the period a disclosure or figure covers.",None,"xsd:dateTime",["B1","B8"]),
("periodEnd","d","period end","End of the period a disclosure or figure covers.",None,"xsd:dateTime",["B1","B8"]),
("costCategory","c","cost category","Category of a cost item.","pwm:CostItem","skos:Concept",["B1"],"CostCategoryScheme"),
("costFromFee","o","cost from fee","Fee that the cost item comes from.","pwm:CostItem","pwm:Fee",["B1"]),
("disclosesPayment","o","discloses payment","Third-party payment that the disclosure describes.","pwm:InducementDisclosure","pwm:ThirdPartyPayment",["B2"]),
("enhancesServiceQuality","d","enhances service quality","Whether the payment is designed to enhance the quality of the service to the client.","pwm:InducementDisclosure","xsd:boolean",["B2"]),
("approvesProduct","o","approves product","Product approved for distribution.","pwm:ProductApproval","pwm:Product",["B3"]),
("approvedForTargetMarket","o","approved for target market","Target market the product is approved for.","pwm:ProductApproval","pwm:TargetMarket",["B3"]),
("distributionStrategyReference","d","distribution strategy reference","Reference to the distribution strategy for the product.","pwm:ProductApproval","xsd:string",["B3"]),
("negativeTargetMarketCategory","c","negative target market category","Investor category the product is not compatible with.","pwm:TargetMarket","skos:Concept",["B3"],"InvestorCategoryScheme"),
("reviewsTargetMarket","o","reviews target market","Target market under review.","pwm:TargetMarketReview","pwm:TargetMarket",["B3"]),
("targetMarketReviewOutcome","c","target market review outcome","Result of the review.","pwm:TargetMarketReview","skos:Concept",["B3"],"TargetMarketReviewOutcomeScheme"),
("nextReviewDueAt","d","next review due at","When the next review is due.",None,"xsd:dateTime",["B3","B5"]),
("complaintOf","o","complaint of","Complaint that the outcome resolves.","pwm:ComplaintOutcome","pwm:Complaint",["B8"]),
("complaintResolution","c","complaint resolution","Result of handling the complaint.","pwm:ComplaintOutcome","skos:Concept",["B8"],"ComplaintResolutionScheme"),
("redressAmount","d","redress amount","Redress paid to the client, in the stated currency.","pwm:ComplaintOutcome","xsd:decimal",["B8"]),
("rootCause","d","root cause","Root cause found for the complaint.","pwm:ComplaintOutcome","xsd:string",["B8"]),
("complaintReceivedAt","d","complaint received at","When the complaint was received.","pwm:Complaint","xsd:dateTime",["B8"]),
("vulnerabilityDriver","c","vulnerability driver","Which of the four drivers of vulnerability applies.","pwm:VulnerabilityIndicator","skos:Concept",["B4"],"VulnerabilityDriverScheme"),
("vulnerabilityDeterminedBy","o","vulnerability determined by","Person who determined the vulnerability. An AI may flag but not determine.","pwm:VulnerabilityIndicator","fibo-fnd-aap-ppl:Person",["B4"]),
("vulnerabilityFlaggedAt","d","vulnerability flagged at","When the indicator was raised.","pwm:VulnerabilityIndicator","xsd:dateTime",["B4"]),
("reviewsSuitability","o","reviews suitability","Suitability assessment under review.","pwm:SuitabilityReview","pwm:SuitabilityAssessment",["B5"]),
("suitabilityReviewResult","c","suitability review result","Result of the periodic review.","pwm:SuitabilityReview","skos:Concept",["B5"],"SuitabilityReviewResultScheme"),
("reviewFrequencyMonths","d","review frequency in months","Agreed frequency of periodic suitability reviews, in months.","pwm:SuitabilityReview","xsd:positiveInteger",["B5"]),
("performedAt","d","performed at","When the review was performed.","pwm:SuitabilityReview","xsd:dateTime",["B5"]),
("orderSide","c","order side","Buy or sell.","pwm:Order","skos:Concept",["B6"],"OrderSideScheme"),
("orderType","c","order type","Market or limit.","pwm:Order","skos:Concept",["B6"],"OrderTypeScheme"),
("orderStatus","c","order status","State of the order.","pwm:Order","skos:Concept",["B6"],"OrderStatusScheme"),
("orderQuantity","d","order quantity","Quantity ordered.","pwm:Order","xsd:decimal",["B6"]),
("limitPrice","d","limit price","Limit price of a limit order.","pwm:Order","xsd:decimal",["B6"]),
("orderAccount","o","order account","Account the order is placed in.","pwm:Order","fibo-fbc-pas-caa:Account",["B6"]),
("orderUnderMandate","o","order under mandate","Mandate the order is placed under.","pwm:Order","pwm:Mandate",["B6"]),
("orderPlacedAt","d","order placed at","When the order was placed.","pwm:Order","xsd:dateTime",["B6"]),
("executesOrder","o","executes order","Order that the execution fills.","pwm:Execution","pwm:Order",["B6"]),
("executedQuantity","d","executed quantity","Quantity filled.","pwm:Execution","xsd:decimal",["B6"]),
("executionPrice","d","execution price","Price of the fill.","pwm:Execution","xsd:decimal",["B6"]),
("executedAt","d","executed at","When the fill happened.","pwm:Execution","xsd:dateTime",["B6"]),
("executionVenue","d","execution venue","Venue where the fill happened.","pwm:Execution","xsd:string",["B6"]),
("allocatesExecution","o","allocates execution","Execution that the allocation assigns.","pwm:Allocation","pwm:Execution",["B6"]),
("allocatedToAccount","o","allocated to account","Account that receives the quantity.","pwm:Allocation","fibo-fbc-pas-caa:Account",["B6"]),
("allocatedQuantity","d","allocated quantity","Quantity assigned to the account.","pwm:Allocation","xsd:decimal",["B6"]),
("bestExecutionOf","o","best execution of","Execution that the record covers.","pwm:BestExecutionRecord","pwm:Execution",["B6"]),
("executionFactor","c","execution factor","Factor considered under the best execution policy.","pwm:BestExecutionRecord","skos:Concept",["B6"],"ExecutionFactorScheme"),
("venueRationale","d","venue rationale","Why the venue was chosen.","pwm:BestExecutionRecord","xsd:string",["B6"]),
("executionPolicyReference","d","execution policy reference","Version of the best execution policy applied.","pwm:BestExecutionRecord","xsd:string",["B6"]),
("corporateActionKind","c","corporate action kind","Kind of corporate action.","pwm:CorporateAction","skos:Concept",["B8"],"CorporateActionKindScheme"),
("affectsInstrument","o","affects instrument","Instrument the action affects.","pwm:CorporateAction","fibo-fbc-fi-fi:FinancialInstrument",["B8"]),
("effectiveAt","d","effective at","When the action takes effect.","pwm:CorporateAction","xsd:dateTime",["B8"]),
("incomeKind","c","income kind","Kind of income.","pwm:IncomeReceipt","skos:Concept",["B8"],"IncomeKindScheme"),
("incomeToAccount","o","income to account","Account credited.","pwm:IncomeReceipt","fibo-fbc-pas-caa:Account",["B8"]),
("incomeFromAction","o","income from action","Corporate action the income comes from.","pwm:IncomeReceipt","pwm:CorporateAction",["B8"]),
("reportingRegime","c","reporting regime","Tax reporting regime.","pwm:TaxReportingRecord","skos:Concept",["B7"],"TaxReportingRegimeScheme"),
("reportingYear","d","reporting year","Calendar year reported.","pwm:TaxReportingRecord","xsd:integer",["B7"]),
("reportedAccount","o","reported account","Account that was reported.","pwm:TaxReportingRecord","fibo-fbc-pas-caa:Account",["B7"]),
("reportedToJurisdiction","d","reported to jurisdiction","Jurisdiction reported to, as an ISO 3166-1 alpha-2 code.","pwm:TaxReportingRecord","xsd:string",["B7"]),
("reportSubmittedAt","d","report submitted at","When the report was submitted.","pwm:TaxReportingRecord","xsd:dateTime",["B7"]),
("facilityKind","c","facility kind","Kind of lending facility.","pwm:LendingFacility","skos:Concept",["B8"],"FacilityKindScheme"),
("facilityLimit","d","facility limit","Credit limit.","pwm:LendingFacility","xsd:decimal",["B8"]),
("maxLoanToValue","d","maximum loan to value","Maximum loan-to-value as a fraction from 0 to 1.","pwm:LendingFacility","xsd:decimal",["B8"]),
("facilityClient","o","facility client","Client who holds the facility.","pwm:LendingFacility","pwm:Client",["B8"]),
("performanceOf","o","performance of","Portfolio that the figure measures.","pwm:PerformanceFigure","pwm:Portfolio",["B8"]),
("performanceMeasure","c","performance measure","Method that produced the figure.","pwm:PerformanceFigure","skos:Concept",["B8"],"PerformanceMeasureScheme"),
("performanceValue","d","performance value","The figure, as a decimal fraction.","pwm:PerformanceFigure","xsd:decimal",["B8"]),
]
S = [
("DisclosureTimingScheme","disclosure timings","MiFID II Art. 24(4)",[("TimingExAnte","ex_ante","ex ante","Before the service is provided."),("TimingExPost","ex_post","ex post","After the period, on the actual costs.")]),
("CostCategoryScheme","cost categories","DR 2017/565 Art. 50",[("CostOneOff","one_off","one-off","One-off charges."),("CostOngoing","ongoing","ongoing","Ongoing charges."),("CostTransaction","transaction","transaction","Transaction costs."),("CostIncidental","incidental","incidental","Incidental costs, such as performance fees."),("CostAncillary","ancillary","ancillary","Ancillary services.")]),
("TargetMarketReviewOutcomeScheme","target market review outcomes","MiFID II Art. 16(3)",[("ReviewConfirmed","confirmed","confirmed","The target market stands."),("ReviewAmended","amended","amended","The target market was changed."),("ReviewWithdrawn","withdrawn","withdrawn","The product was withdrawn from distribution.")]),
("ComplaintResolutionScheme","complaint resolutions","PWM",[("ResolutionUpheld","upheld","upheld","Upheld."),("ResolutionPartlyUpheld","partially_upheld","partially upheld","Partially upheld."),("ResolutionNotUpheld","not_upheld","not upheld","Not upheld."),("ResolutionWithdrawn","withdrawn","withdrawn by client","Withdrawn by the client.")]),
("VulnerabilityDriverScheme","vulnerability drivers","FCA FG21/1",[("DriverHealth","health","health","Health conditions that affect the ability to make decisions."),("DriverLifeEvent","life_events","life events","Life events such as bereavement."),("DriverResilience","resilience","resilience","Low ability to withstand financial or emotional shocks."),("DriverCapability","capability","capability","Low knowledge of financial matters or low digital skills.")]),
("SuitabilityReviewResultScheme","suitability review results","MiFID II Art. 25(2)",[("SuitabilityStill","still_suitable","still suitable","The recommendation remains suitable."),("SuitabilityNo","no_longer_suitable","no longer suitable","The recommendation is no longer suitable."),("SuitabilityUpdate","update_required","update required","The assessment needs updating.")]),
("OrderSideScheme","order sides","PWM",[("SideBuy","buy","buy","Buy."),("SideSell","sell","sell","Sell.")]),
("OrderTypeScheme","order types","PWM",[("TypeMarket","market","market","Market order."),("TypeLimit","limit","limit","Limit order.")]),
("OrderStatusScheme","order statuses","PWM",[("OrderPlaced","placed","placed","Placed."),("OrderPartlyFilled","partially_filled","partially filled","Partially filled."),("OrderFilled","filled","filled","Filled."),("OrderCanceled","canceled","canceled","Canceled."),("OrderRejected","rejected","rejected","Rejected.")]),
("ExecutionFactorScheme","best execution factors","MiFID II Art. 27(1)",[("FactorPrice","price","price","Price."),("FactorCosts","costs","costs","Costs."),("FactorSpeed","speed","speed","Speed."),("FactorLikelihood","likelihood","likelihood of execution and settlement","Likelihood of execution and settlement."),("FactorSize","size","size","Size."),("FactorNature","nature","nature","Nature of the order.")]),
("CorporateActionKindScheme","corporate action kinds","PWM",[("ActionDividend","dividend","dividend","Dividend."),("ActionCoupon","coupon","coupon","Coupon payment."),("ActionSplit","split","split","Split."),("ActionMerger","merger","merger","Merger."),("ActionRightsIssue","rights_issue","rights issue","Rights issue."),("ActionRedemption","redemption","redemption","Redemption.")]),
("IncomeKindScheme","income kinds","PWM",[("IncomeDividend","dividend","dividend","Dividend income."),("IncomeInterest","interest","interest","Interest income."),("IncomeCoupon","coupon","coupon","Coupon income.")]),
("TaxReportingRegimeScheme","tax reporting regimes","OECD CRS; FATCA",[("RegimeCRS","crs","CRS","OECD Common Reporting Standard."),("RegimeFATCA","fatca","FATCA","US Foreign Account Tax Compliance Act.")]),
("FacilityKindScheme","lending facility kinds","PWM",[("FacilityLombard","lombard","Lombard","Lombard loan against a securities portfolio."),("FacilityMortgage","mortgage","mortgage","Mortgage."),("FacilityStructured","structured_loan","structured loan","Structured loan."),("FacilityOther","other","other","Other credit.")]),
("PerformanceMeasureScheme","performance measures","PWM",[("MeasureTWR","time_weighted_return","time-weighted return","Time-weighted return."),("MeasureMWR","money_weighted_return","money-weighted return","Money-weighted return."),("MeasureAbsolute","absolute_return","absolute return","Absolute return.")]),
]
g.C, g.P, g.S = C, P, S
def module_ttl():
    o=["# =============================================================================",
       "#  PWM 0.10: Business scope (module 13)",
       "#  Costs and charges, product governance, client circumstances, operations and",
       "#  lifecycle, and performance provenance. Generated by tools/gen_module13.py.",
       "#  PWM records these; calculations stay in registered calculation services.",
       "# =============================================================================","",
       'pwm:ModuleBusiness a pwm:Module ; rdfs:label "Business scope"@en ; pwm:moduleOrder 13 ;',
       '    skos:definition "Costs and charges, product governance, client circumstances, orders and executions, corporate actions and income, tax reporting, lending and performance provenance."@en .',""]
    for n,l,d,par,sp,cl,locs in C:
        o.append(f'pwm:{n} a owl:Class ; rdfs:label "{l}"@en ;\n    skos:definition "{d}"@en ; rdfs:subClassOf {par} ;\n    pwm:inModule pwm:ModuleBusiness ; pwm:contextSpeed pwm:{sp}Context ; pwm:defaultClassification pwm:{cl} ;\n    pwm:citation {cit(*locs)} .\n')
    for p in P:
        n,k,l,d,dom,rng,locs=p[:7]
        t="owl:ObjectProperty" if k in "oc" else "owl:DatatypeProperty"
        s=f'pwm:{n} a {t} ; rdfs:label "{l}"@en ;\n    skos:definition "{d}"@en ;'
        if dom: s+=f'\n    rdfs:domain {dom} ;'
        s+=f'\n    rdfs:range {rng} ;\n    pwm:inModule pwm:ModuleBusiness ;\n    pwm:citation {cit(*locs)} .\n'
        o.append(s)
    for sn,sl,scope,cs in S:
        o.append(f'pwm:{sn} a skos:ConceptScheme ;\n    skos:prefLabel "{sl}"@en ;\n    pwm:inModule pwm:ModuleBusiness ;\n    pwm:citation [ pwm:cites pwmref:PWM-PRACTICE ; pwm:locator "0.10 business scope; grounds: {scope}" ] .')
        for cn,nt,lab,df in cs:
            o.append(f'pwm:{cn} a skos:Concept ; skos:notation "{nt}" ; skos:inScheme pwm:{sn} ; skos:topConceptOf pwm:{sn} ; skos:prefLabel "{lab}"@en ; skos:definition "{df}"@en .')
        o.append("")
    o.append("# A Lombard facility is one kind of lending facility.\npwm:LombardFacility rdfs:subClassOf pwm:LendingFacility .")
    return "\n".join(o)+"\n"
def bindings(): return "\n".join(f"pwm:{p[0]} pwm:valuesFrom pwm:{p[7]} ." for p in P if p[1]=="c" and p[0]!="negativeTargetMarketCategory")+"\npwm:negativeTargetMarketCategory pwm:valuesFrom pwm:InvestorCategoryScheme .\n"
SH=[]
def add(name,target,body,locs=("KOG-ROADMAP",)):
    SH.append(g.node(name,target,locs,body))
prop,S1,sparql=g.prop,g.S1,g.sparql
DT="xsd:dateTime"; DEC="xsd:decimal"
add("CostDisclosure","pwm:CostDisclosure",[prop("disclosureTiming",1,1),prop("disclosedToClient",1,1,None,"pwm:Client"),prop("disclosedAt",1,1,DT),prop("coversMandate",1,1,None,"pwm:Mandate"),
 prop("totalCostAmount",1,1,DEC),prop("currency",1,1),prop("hasCostItem",1,None,None,"pwm:CostItem"),prop("includesInducements",1,1,"xsd:boolean"),
 sparql("An ex post disclosure states the period it covers (MiFID II Art. 24(4); Delegated Regulation 2017/565 Art. 50).",
  "            SELECT $this WHERE {\n                $this pwm:disclosureTiming pwm:TimingExPost .\n                FILTER NOT EXISTS { $this pwm:periodStart ?s . $this pwm:periodEnd ?e }\n            }")],("EU-MIFID2","PWM-PRACTICE"))
add("CostItem","pwm:CostItem",[prop("costCategory",1,1),prop("amount",1,1,DEC),prop("currency",1,1)],("EU-MIFID2","PWM-PRACTICE"))
add("InducementDisclosure","pwm:InducementDisclosure",[prop("disclosesPayment",1,1,None,"pwm:ThirdPartyPayment"),prop("disclosedToClient",1,1,None,"pwm:Client"),prop("disclosedAt",1,1,DT),prop("enhancesServiceQuality",1,1,"xsd:boolean")],("EU-MIFID2","PWM-PRACTICE"))
add("ProductApproval","pwm:ProductApproval",[prop("approvesProduct",1,1,None,"pwm:Product"),prop("approvedForTargetMarket",1,1,None,"pwm:TargetMarket"),S1("distributionStrategyReference"),prop("approvedBy",1,1),prop("generatedAtTime",1,1,DT).replace("pwm:generatedAtTime","prov:generatedAtTime")],("EU-MIFID2","PWM-PRACTICE"))
add("TargetMarketReview","pwm:TargetMarketReview",[prop("reviewsTargetMarket",1,1,None,"pwm:TargetMarket"),prop("targetMarketReviewOutcome",1,1),prop("reviewedAt",1,1,DT,extra=("sh:lessThan pwm:nextReviewDueAt","before the next review date")),prop("nextReviewDueAt",1,1,DT)],("EU-MIFID2","PWM-PRACTICE"))
add("ComplaintOutcome","pwm:ComplaintOutcome",[prop("complaintOf",1,1,None,"pwm:Complaint"),prop("complaintResolution",1,1),prop("reviewedAt",1,1,DT),
 sparql("Redress needs a currency (Kognita record of the outcome).","            SELECT $this WHERE {\n                $this pwm:redressAmount ?r .\n                FILTER NOT EXISTS { $this pwm:currency ?c }\n            }")])
add("SuitabilityReview","pwm:SuitabilityReview",[prop("reviewsSuitability",1,1,None,"pwm:SuitabilityAssessment"),prop("suitabilityReviewResult",1,1),prop("reviewFrequencyMonths",1,1,"xsd:positiveInteger"),
 prop("performedAt",1,1,DT,extra=("sh:lessThan pwm:nextReviewDueAt","before the next review date")),prop("nextReviewDueAt",1,1,DT)],("EU-MIFID2","PWM-PRACTICE"))
add("Order","pwm:Order",[prop("orderSide",1,1),prop("orderType",1,1),prop("orderStatus",1,1),prop("orderQuantity",1,1,DEC),prop("inInstrument",1,1,None,"fibo-fbc-fi-fi:FinancialInstrument"),prop("orderAccount",1,1,None,"fibo-fbc-pas-caa:Account"),prop("orderPlacedAt",1,1,DT),
 sparql("A limit order states its limit price.","            SELECT $this WHERE {\n                $this pwm:orderType pwm:TypeLimit .\n                FILTER NOT EXISTS { $this pwm:limitPrice ?p }\n            }")],("EU-MIFID2","PWM-PRACTICE"))
add("Execution","pwm:Execution",[prop("executesOrder",1,1,None,"pwm:Order"),prop("executedQuantity",1,1,DEC),prop("executionPrice",1,1,DEC),prop("executedAt",1,1,DT),S1("executionVenue"),
 sparql("A fill cannot exceed the order quantity.","            SELECT $this WHERE {\n                $this pwm:executesOrder ?o ; pwm:executedQuantity ?q .\n                ?o pwm:orderQuantity ?max .\n                FILTER (?q > ?max)\n            }"),
 sparql("Allocations of one execution cannot add up to more than was executed.","            SELECT $this WHERE {\n                $this pwm:executedQuantity ?q .\n                { SELECT $this (SUM(?a) AS ?total) WHERE { ?x pwm:allocatesExecution $this ; pwm:allocatedQuantity ?a } GROUP BY $this }\n                FILTER (?total > ?q)\n            }"),
 sparql("A fill cannot happen before the order was placed.","            SELECT $this WHERE {\n                $this pwm:executesOrder ?o ; pwm:executedAt ?t .\n                ?o pwm:orderPlacedAt ?p .\n                FILTER (?t < ?p)\n            }")],("EU-MIFID2","PWM-PRACTICE"))
add("Allocation","pwm:Allocation",[prop("allocatesExecution",1,1,None,"pwm:Execution"),prop("allocatedToAccount",1,1,None,"fibo-fbc-pas-caa:Account"),prop("allocatedQuantity",1,1,DEC)],("EU-MIFID2","PWM-PRACTICE"))
add("BestExecutionRecord","pwm:BestExecutionRecord",[prop("bestExecutionOf",1,1,None,"pwm:Execution"),prop("executionFactor",1,None),S1("venueRationale"),S1("executionPolicyReference")],("EU-MIFID2","PWM-PRACTICE"))
add("CorporateAction","pwm:CorporateAction",[prop("corporateActionKind",1,1),prop("affectsInstrument",1,1,None,"fibo-fbc-fi-fi:FinancialInstrument"),prop("effectiveAt",1,1,DT)])
add("IncomeReceipt","pwm:IncomeReceipt",[prop("incomeKind",1,1),prop("incomeToAccount",1,1,None,"fibo-fbc-pas-caa:Account"),prop("amount",1,1,DEC),prop("currency",1,1)])
add("TaxReportingRecord","pwm:TaxReportingRecord",[prop("reportingRegime",1,1),prop("reportingYear",1,1,"xsd:integer"),prop("reportedAccount",1,1,None,"fibo-fbc-pas-caa:Account"),
 "    sh:property [ sh:path pwm:reportedToJurisdiction ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:string ; sh:pattern \"^[A-Z]{2}$\" ; sh:message \"pwm:reportedToJurisdiction: an ISO 3166-1 alpha-2 code.\" ]",
 prop("amount",1,1,DEC),prop("currency",1,1),prop("reportSubmittedAt",1,1,DT)],("OECD-CRS","PWM-PRACTICE"))
add("LendingFacility","pwm:LendingFacility",[prop("facilityKind",1,1),prop("facilityClient",1,1,None,"pwm:Client"),prop("facilityLimit",1,1,DEC),prop("currency",1,1),
 "    sh:property [ sh:path pwm:maxLoanToValue ; sh:minCount 1 ; sh:maxCount 1 ; sh:datatype xsd:decimal ; sh:minInclusive 0 ; sh:maxInclusive 1 ; sh:message \"pwm:maxLoanToValue: between 0 and 1.\" ]"])
add("PerformanceFigure","pwm:PerformanceFigure",[prop("performanceOf",1,1,None,"pwm:Portfolio"),prop("performanceMeasure",1,1),prop("performanceValue",1,1,DEC),
 prop("periodStart",1,1,DT,extra=("sh:lessThan pwm:periodEnd","before periodEnd")),prop("periodEnd",1,1,DT),
 "    sh:property [ sh:path pwm:calculatedBy ; sh:minCount 1 ; sh:class pwm:CalculationService ; sh:message \"A performance figure cites a registered calculation service; PWM never computes it (Kognita 0.6 item 6).\" ]"])
def shapes_text(): return "\n".join(SH)
POS='''
# --- Module 13: business scope (synthetic) ------------------------------------------------------
ex:CostDisclosure1 a pwm:CostDisclosure ; pwm:disclosureTiming pwm:TimingExAnte ; pwm:disclosedToClient ex:Client ; pwm:disclosedAt "2026-11-02T10:00:00+08:00"^^xsd:dateTime ; pwm:coversMandate ex:Mandate ; pwm:totalCostAmount "1200.00"^^xsd:decimal ; pwm:totalCostPercent "1.2"^^xsd:decimal ; pwm:currency "SGD" ; pwm:includesInducements true ; pwm:hasCostItem ex:CostItem1, ex:CostItem2 .
ex:CostDisclosure2 a pwm:CostDisclosure ; pwm:disclosureTiming pwm:TimingExPost ; pwm:disclosedToClient ex:Client ; pwm:disclosedAt "2026-12-31T10:00:00+08:00"^^xsd:dateTime ; pwm:coversMandate ex:Mandate ; pwm:totalCostAmount "1180.00"^^xsd:decimal ; pwm:currency "SGD" ; pwm:includesInducements false ; pwm:periodStart "2026-01-01T00:00:00+08:00"^^xsd:dateTime ; pwm:periodEnd "2026-12-31T00:00:00+08:00"^^xsd:dateTime ; pwm:hasCostItem ex:CostItem1 .
ex:CostItem1 a pwm:CostItem ; pwm:costCategory pwm:CostOngoing ; pwm:amount "1000.00"^^xsd:decimal ; pwm:currency "SGD" ; pwm:costFromFee ex:Fee1 .
ex:CostItem2 a pwm:CostItem ; pwm:costCategory pwm:CostTransaction ; pwm:amount "200.00"^^xsd:decimal ; pwm:currency "SGD" .
ex:InducementDisclosure1 a pwm:InducementDisclosure ; pwm:disclosesPayment ex:ThirdPartyPayment1 ; pwm:disclosedToClient ex:Client ; pwm:disclosedAt "2026-11-02T10:00:00+08:00"^^xsd:dateTime ; pwm:enhancesServiceQuality true .
ex:ProductApproval1 a pwm:ProductApproval ; pwm:approvesProduct ex:Product ; pwm:approvedForTargetMarket ex:TargetMarket1 ; pwm:distributionStrategyReference "urn:synthetic:distribution-strategy:0001" ; pwm:approvedBy ex:CommitteeDG ; prov:generatedAtTime "2026-09-01T00:00:00+08:00"^^xsd:dateTime .
ex:CommitteeDG a pwm:DataGovernanceCommittee ; rdfs:label "Approving committee (fictional)"@en .
ex:TargetMarket1 pwm:negativeTargetMarketCategory pwm:RetailInvestorSG .
ex:TargetMarketReview1 a pwm:TargetMarketReview ; pwm:reviewsTargetMarket ex:TargetMarket1 ; pwm:targetMarketReviewOutcome pwm:ReviewConfirmed ; pwm:reviewedAt "2026-10-01T00:00:00+08:00"^^xsd:dateTime ; pwm:nextReviewDueAt "2027-10-01T00:00:00+08:00"^^xsd:dateTime .
ex:Complaint1 pwm:complaintReceivedAt "2026-10-20T09:00:00+08:00"^^xsd:dateTime .
ex:ComplaintOutcome1 a pwm:ComplaintOutcome ; pwm:complaintOf ex:Complaint1 ; pwm:complaintResolution pwm:ResolutionPartlyUpheld ; pwm:reviewedAt "2026-11-01T00:00:00+08:00"^^xsd:dateTime ; pwm:redressAmount "50.00"^^xsd:decimal ; pwm:currency "SGD" ; pwm:rootCause "Late statement run" .
ex:Vulnerability1 pwm:vulnerabilityDriver pwm:DriverLifeEvent ; pwm:vulnerabilityDeterminedBy ex:PersonRM ; pwm:vulnerabilityFlaggedAt "2026-10-25T09:00:00+08:00"^^xsd:dateTime .
ex:SuitabilityReview1 a pwm:SuitabilityReview ; pwm:reviewsSuitability ex:Suitability ; pwm:suitabilityReviewResult pwm:SuitabilityStill ; pwm:reviewFrequencyMonths "12"^^xsd:positiveInteger ; pwm:performedAt "2026-11-02T00:00:00+08:00"^^xsd:dateTime ; pwm:nextReviewDueAt "2027-11-02T00:00:00+08:00"^^xsd:dateTime .
ex:Order1 a pwm:Order ; pwm:orderSide pwm:SideBuy ; pwm:orderType pwm:TypeLimit ; pwm:orderStatus pwm:OrderFilled ; pwm:orderQuantity "100"^^xsd:decimal ; pwm:limitPrice "99.50"^^xsd:decimal ; pwm:inInstrument ex:Bond ; pwm:orderAccount ex:Account ; pwm:orderUnderMandate ex:Mandate ; pwm:orderPlacedAt "2026-11-03T11:00:00+08:00"^^xsd:dateTime .
ex:Execution1 a pwm:Execution ; pwm:executesOrder ex:Order1 ; pwm:executedQuantity "100"^^xsd:decimal ; pwm:executionPrice "99.40"^^xsd:decimal ; pwm:executedAt "2026-11-03T11:05:00+08:00"^^xsd:dateTime ; pwm:executionVenue "Synthetic venue" .
ex:Allocation1 a pwm:Allocation ; pwm:allocatesExecution ex:Execution1 ; pwm:allocatedToAccount ex:Account ; pwm:allocatedQuantity "100"^^xsd:decimal .
ex:BestExecution1 a pwm:BestExecutionRecord ; pwm:bestExecutionOf ex:Execution1 ; pwm:executionFactor pwm:FactorPrice, pwm:FactorCosts, pwm:FactorLikelihood ; pwm:venueRationale "Best displayed price with settlement certainty" ; pwm:executionPolicyReference "best-execution-policy-2026-1" .
ex:CorporateAction1 a pwm:CorporateAction ; pwm:corporateActionKind pwm:ActionCoupon ; pwm:affectsInstrument ex:Bond ; pwm:effectiveAt "2026-12-15T00:00:00+08:00"^^xsd:dateTime .
ex:Income1 a pwm:IncomeReceipt ; pwm:incomeKind pwm:IncomeCoupon ; pwm:incomeToAccount ex:Account ; pwm:incomeFromAction ex:CorporateAction1 ; pwm:amount "2500.00"^^xsd:decimal ; pwm:currency "SGD" .
ex:TaxReport1 a pwm:TaxReportingRecord ; pwm:reportingRegime pwm:RegimeCRS ; pwm:reportingYear "2026"^^xsd:integer ; pwm:reportedAccount ex:Account ; pwm:reportedToJurisdiction "SG" ; pwm:amount "1000000.00"^^xsd:decimal ; pwm:currency "SGD" ; pwm:reportSubmittedAt "2027-05-01T00:00:00+08:00"^^xsd:dateTime .
ex:Lombard1 pwm:facilityKind pwm:FacilityLombard ; pwm:facilityClient ex:Client ; pwm:facilityLimit "500000.00"^^xsd:decimal ; pwm:currency "SGD" ; pwm:maxLoanToValue "0.50"^^xsd:decimal .
ex:Performance1 a pwm:PerformanceFigure ; pwm:performanceOf ex:Portfolio ; pwm:performanceMeasure pwm:MeasureTWR ; pwm:performanceValue "0.042"^^xsd:decimal ; pwm:periodStart "2026-01-01T00:00:00+08:00"^^xsd:dateTime ; pwm:periodEnd "2026-10-31T00:00:00+08:00"^^xsd:dateTime ; pwm:calculatedBy ex:ValuationService ; pwm:hasBenchmark ex:Benchmark1 .
'''
BAD=[
("BadBizCostDisclosure","CostDisclosureShape","currency",'a pwm:CostDisclosure ; pwm:disclosureTiming pwm:TimingExAnte ; pwm:disclosedToClient ex:Client ; pwm:disclosedAt "2026-11-02T10:00:00+08:00"^^xsd:dateTime ; pwm:coversMandate ex:Mandate ; pwm:totalCostAmount "10"^^xsd:decimal ; pwm:includesInducements false ; pwm:hasCostItem ex:CostItem1'),
("BadBizExPost","CostDisclosureShape",None,'a pwm:CostDisclosure ; pwm:disclosureTiming pwm:TimingExPost ; pwm:disclosedToClient ex:Client ; pwm:disclosedAt "2026-12-31T10:00:00+08:00"^^xsd:dateTime ; pwm:coversMandate ex:Mandate ; pwm:totalCostAmount "10"^^xsd:decimal ; pwm:currency "SGD" ; pwm:includesInducements false ; pwm:hasCostItem ex:CostItem1'),
("BadBizCostItem","CostItemShape","costCategory",'a pwm:CostItem ; pwm:amount "5"^^xsd:decimal ; pwm:currency "SGD"'),
("BadBizInducement","InducementDisclosureShape","enhancesServiceQuality",'a pwm:InducementDisclosure ; pwm:disclosesPayment ex:ThirdPartyPayment1 ; pwm:disclosedToClient ex:Client ; pwm:disclosedAt "2026-11-02T10:00:00+08:00"^^xsd:dateTime'),
("BadBizProductApproval","ProductApprovalShape","distributionStrategyReference",'a pwm:ProductApproval ; pwm:approvesProduct ex:Product ; pwm:approvedForTargetMarket ex:TargetMarket1 ; pwm:approvedBy ex:CommitteeDG ; prov:generatedAtTime "2026-09-01T00:00:00+08:00"^^xsd:dateTime'),
("BadBizTargetMarketReview","TargetMarketReviewShape","reviewedAt",'a pwm:TargetMarketReview ; pwm:reviewsTargetMarket ex:TargetMarket1 ; pwm:targetMarketReviewOutcome pwm:ReviewConfirmed ; pwm:reviewedAt "2027-10-01T00:00:00+08:00"^^xsd:dateTime ; pwm:nextReviewDueAt "2026-10-01T00:00:00+08:00"^^xsd:dateTime'),
("BadBizComplaintOutcome","ComplaintOutcomeShape",None,'a pwm:ComplaintOutcome ; pwm:complaintOf ex:Complaint1 ; pwm:complaintResolution pwm:ResolutionUpheld ; pwm:reviewedAt "2026-11-01T00:00:00+08:00"^^xsd:dateTime ; pwm:redressAmount "50.00"^^xsd:decimal'),
("BadBizSuitabilityReview","SuitabilityReviewShape","performedAt",'a pwm:SuitabilityReview ; pwm:reviewsSuitability ex:Suitability ; pwm:suitabilityReviewResult pwm:SuitabilityNo ; pwm:reviewFrequencyMonths "12"^^xsd:positiveInteger ; pwm:performedAt "2028-01-01T00:00:00+08:00"^^xsd:dateTime ; pwm:nextReviewDueAt "2027-11-02T00:00:00+08:00"^^xsd:dateTime'),
("BadBizLimitOrder","OrderShape",None,'a pwm:Order ; pwm:orderSide pwm:SideSell ; pwm:orderType pwm:TypeLimit ; pwm:orderStatus pwm:OrderPlaced ; pwm:orderQuantity "10"^^xsd:decimal ; pwm:inInstrument ex:Bond ; pwm:orderAccount ex:Account ; pwm:orderPlacedAt "2026-11-03T11:00:00+08:00"^^xsd:dateTime'),
("BadBizOverfill","ExecutionShape",None,'a pwm:Execution ; pwm:executesOrder ex:Order1 ; pwm:executedQuantity "150"^^xsd:decimal ; pwm:executionPrice "99.40"^^xsd:decimal ; pwm:executedAt "2026-11-03T11:06:00+08:00"^^xsd:dateTime ; pwm:executionVenue "Synthetic venue"'),
("BadBizEarlyFill","ExecutionShape",None,'a pwm:Execution ; pwm:executesOrder ex:Order1 ; pwm:executedQuantity "10"^^xsd:decimal ; pwm:executionPrice "99.40"^^xsd:decimal ; pwm:executedAt "2026-11-03T10:00:00+08:00"^^xsd:dateTime ; pwm:executionVenue "Synthetic venue"'),
("BadBizOverAllocation","ExecutionShape",None,'a pwm:Allocation ; pwm:allocatesExecution ex:Execution1 ; pwm:allocatedToAccount ex:Account ; pwm:allocatedQuantity "50"^^xsd:decimal'),
("BadBizAllocation","AllocationShape","allocatedQuantity",'a pwm:Allocation ; pwm:allocatesExecution ex:Execution1 ; pwm:allocatedToAccount ex:Account'),
("BadBizBestExecution","BestExecutionRecordShape","venueRationale",'a pwm:BestExecutionRecord ; pwm:bestExecutionOf ex:Execution1 ; pwm:executionFactor pwm:FactorPrice ; pwm:executionPolicyReference "p-1"'),
("BadBizCorporateAction","CorporateActionShape","effectiveAt",'a pwm:CorporateAction ; pwm:corporateActionKind pwm:ActionSplit ; pwm:affectsInstrument ex:Bond'),
("BadBizIncome","IncomeReceiptShape","incomeToAccount",'a pwm:IncomeReceipt ; pwm:incomeKind pwm:IncomeInterest ; pwm:amount "1"^^xsd:decimal ; pwm:currency "SGD"'),
("BadBizTaxReport","TaxReportingRecordShape","reportedToJurisdiction",'a pwm:TaxReportingRecord ; pwm:reportingRegime pwm:RegimeFATCA ; pwm:reportingYear "2026"^^xsd:integer ; pwm:reportedAccount ex:Account ; pwm:reportedToJurisdiction "usa" ; pwm:amount "1"^^xsd:decimal ; pwm:currency "USD" ; pwm:reportSubmittedAt "2027-03-01T00:00:00+08:00"^^xsd:dateTime'),
("BadBizFacility","LendingFacilityShape","maxLoanToValue",'a pwm:LendingFacility ; pwm:facilityKind pwm:FacilityMortgage ; pwm:facilityClient ex:Client ; pwm:facilityLimit "1"^^xsd:decimal ; pwm:currency "SGD" ; pwm:maxLoanToValue "1.5"^^xsd:decimal'),
("BadBizPerformance","PerformanceFigureShape","calculatedBy",'a pwm:PerformanceFigure ; pwm:performanceOf ex:Portfolio ; pwm:performanceMeasure pwm:MeasureMWR ; pwm:performanceValue "0.01"^^xsd:decimal ; pwm:periodStart "2026-01-01T00:00:00+08:00"^^xsd:dateTime ; pwm:periodEnd "2026-10-31T00:00:00+08:00"^^xsd:dateTime'),
("BadBizPerformancePeriod","PerformanceFigureShape","periodStart",'a pwm:PerformanceFigure ; pwm:performanceOf ex:Portfolio ; pwm:performanceMeasure pwm:MeasureMWR ; pwm:performanceValue "0.01"^^xsd:decimal ; pwm:periodStart "2026-12-01T00:00:00+08:00"^^xsd:dateTime ; pwm:periodEnd "2026-10-31T00:00:00+08:00"^^xsd:dateTime ; pwm:calculatedBy ex:ValuationService'),
]
if __name__=="__main__":
    B,E="# >>> module 13 BEGIN (generated by tools/gen_module13.py)","# <<< module 13 END"
    def put(path,block):
        f=ROOT/path; t=f.read_text()
        t=re.sub(r"\n+"+re.escape(B)+r".*?"+re.escape(E)+r"\n","\n",t,flags=re.S)
        f.write_text(t.rstrip("\n")+"\n\n"+B+"\n"+block+E+"\n")
    (ROOT/"src/13-business-scope.ttl").write_text(module_ttl())
    put("src/08-code-list-bindings.ttl",bindings()); put("src/shapes/pwm-shapes.ttl",shapes_text())
    put("examples/pwm-example.ttl",POS)
    h="7"*64
    put("examples/pwm-example-invalid.ttl","\n".join(f"ex:{n} {t} ." for n,s,p,t in BAD)+"\n")
    exp=[(('Execution1' if n=='BadBizOverAllocation' else n),s,p) for n,s,p,t in BAD]
    vf=ROOT/"tools/validate.py"; t=vf.read_text()
    t=re.sub(r"\n+"+re.escape(B)+r".*?"+re.escape(E)+r"\n","\n",t,flags=re.S)
    t=t.replace("\ndef local(term):","\n"+B+"\nEXPECTED.update("+repr(exp)+")\n"+E+"\n\ndef local(term):",1)
    vf.write_text(t)
    print("generated",len(C),"classes",len(P),"properties",len(S),"schemes",len(BAD),"bad fixtures")
