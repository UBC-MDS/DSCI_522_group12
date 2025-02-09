## Feedback from Milestones 1-3

### Change 1

**Feedback:** "You should mention the findings of your analysis in the README file."  
**Change:** Added the findings of the analysis including the final results, the limitations and further directions.  
**Link:** https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/blob/main/README.md  
**Commit:** https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/commit/9d2d3e38471447abad8ab2687ec4fb3dec8145d8

### Change 2
**Feedback:** "No Creative Commons license (for project report) was specified."  
**Change:** Added the CC license for the project report in the License file.  
**Link:** https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/blob/main/LICENSE.md  
**Commit:** https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/commit/10be574322789e64d74c3e3f2653a4f222751e98#diff-4673a3aba01813b595de187a7a6e9e63a3491d55821606fecd9f13a10c188a1d

### Change 3

**Feedback:** "What does it mean by customer satisfaction? Please state your question more clearly. Limitations of the findings were not stated (all studies have limitations)."  
**Change:** Elaborated more on the term "customer satisfaction". Limitations are added to the summary section.  
**Link:** https://ubc-mds.github.io/airline-customer-satisfaction-predictor/docs/airline_passenger_satisfaction_predictor.html  
**Commit:** https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/commit/9098d1e77b65e22e96e0fe414f334d0ab6c1ba02#diff-5b01df08ec4960af7c5352e5f86f8b41803f89b15e6626c055d5116a35f5ea45

### Change 4

**Feedback:** "Your introduction is far too brief. Please be more specific on what customer satisfaction is and what it measures - is it a categorical value, or a numeric score?"  
**Change:** Modified the introduction to elaborate more on the target variable. Additionally, added high-level dataset description.  
**Link:** https://ubc-mds.github.io/airline-customer-satisfaction-predictor/docs/airline_passenger_satisfaction_predictor.html  
**Commit:** https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/commit/ff02dbc50cd5931fd961016302905f44c3a7e696#diff-9f4005064ec0c6c74aaee0d0e0b2651b521b3ba7936ed527ac2785b0a6e5adc6

### Change 5

**Feedback:** "Please use 'Results' as the header - not 'Reporting Performance'. Also please include figure descriptions under your figures (should be 1-2 sentences to describe the figure for readers who only read the figure)."  
**Change:** Results header is added. Meaningful figure descriptions are also added.
**Link:** https://ubc-mds.github.io/airline-customer-satisfaction-predictor/docs/airline_passenger_satisfaction_predictor.html  
**Commit:** https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/commit/8cb565bc8a5c1175e70745a1a79795759070e85d#diff-9f4005064ec0c6c74aaee0d0e0b2651b521b3ba7936ed527ac2785b0a6e5adc6

### Change 6

**Feedback:** "Can you mention some assumptions and limitations of the methods and your findings? Also, please reiterate some key findings in this part of your report."  
**Change:** Limitations and assumptions, as well as key findings are added. 
**Link:** https://ubc-mds.github.io/airline-customer-satisfaction-predictor/docs/airline_passenger_satisfaction_predictor.html  

### Change 7

**Feedback:** "If there are acronyms in your report, you should explain what they mean (ROC, SHAP, ...). Also try to avoid bullet points in your report."  
**Change:** Acronyms are removed, instead, full titles are used. Unnecessary bullet points are also removed.  
**Link:** https://ubc-mds.github.io/airline-customer-satisfaction-predictor/docs/airline_passenger_satisfaction_predictor.html  

### Change 8

**Feedback:** "For data validation checks, you throw errors for most checks, but for duplicates you are just printing a warning. The behaviour should be the same for all. -0.5
The checks for anomalous correlations between features, and targets and features are just visual checks as far as I can see. They should be programatic and do something (like the pandera checks). 2 * -0.5"  
**Change:** All checks now throw an error.  
**Link:** https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/blob/review-changes/src/data_validation_utils.py  


### Change 9

**Feedback:** "There was a typo which caused the volume mounting to be incorrect."  
**Change:** The typo was corrected. Docker works fine.  
**Link:** https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/commit/e779ffb00ed022625d7bea7d0e3f9e315e12d35e  
**Commit:** https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/commit/e779ffb00ed022625d7bea7d0e3f9e315e12d35e  

### Change 10

**Feedback:** "[Reviewer: Hui Tang]: While the decision tree model is clearly explained in terms of what it does, it would be helpful to justify why you chose a decision tree over other classification methods. A brief comparison or rationale would strengthen the methods section and showcase understanding of model selection criteria."  
**Change:** Added justification why decision tree model was used in the Analysis section.  
**Link:** https://ubc-mds.github.io/airline-customer-satisfaction-predictor/docs/airline_passenger_satisfaction_predictor.html  

### Change 11

**Feedback:** "[Reviewer: Hui Tang]: Additional Testing and Validation:
The tests included are helpful, but expanding test coverage—particularly including tests for edge cases or ensuring all scripts run smoothly under unexpected input conditions—would increase confidence in the robustness of the analysis pipeline."  
**Change:** Included robust test cases: for expected, error, and edge cases.  
**Link:** https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/tree/main/tests  

### Change 12

**Feedback:** No feedback.  
**Change:** Updated the readme to include the make all and make clean commands. Removed unnecessary content.  
**Link:** https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/blob/main/README.md    

### Other

All other feedbacks were either not applicable to the final milestone or were not approved by our team to be a valid change.

## Future Tasks
Some future considerations for the project include: adding model interpretation analysis through feature importance to imporve transparency (Thanks Chaoyu Ou and all other reviewers for mentioning it).
