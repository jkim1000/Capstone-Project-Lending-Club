# Lending Club
Lending Club default prediction

**#Introduction**

In financial industry, banks have historically handled most consumer and small business lending to a great extent. However, banks have some key limitations like interest rates are not individualized, loan decisions can take months, regulation process is strong and the costs of underwriting loans are high. Peer-to-peer (P2P) lending platforms offer loan opportunities outside of traditional lending institutions and it generally provides higher returns to the investors relative to other types of investments because of the lower regulatory burdens.

Founded in 2007, Lending Club pioneered an online peer to peer marketplace that sought to streamline the process of obtaining a loan by directly connecting borrowers and lenders without an intermediary. Borrowers would fill out an application detailing their credit history, loan details, employment status and other self-reported information by which Lending Club would assign a loan grade reflecting the quality of a loan. Potential investors would be able to view the details of various loan applications and make a final decision to invest in the loan. Lending Club would take a flat service fee from the transaction and this was much preferable to traditional bank loans since banks would take part in the interest payments. 


![image](https://user-images.githubusercontent.com/67875208/118530339-88f53e00-b712-11eb-9235-b28e620156c2.png)

However, with any type of loan, risks of default are major concerns for the investors. Most loans that were listed on Lending Club were unsecured personal loans, meaning that there was no collateral backing up the loan. Therefore, the lender/investor would essentially lose all their money if the borrower defaulted. Such financial losses incurred by default loans are detrimental to both borrowers and especially investors, which is why a machine learning approach can be beneficial to help filter out bad loans and create a more risk-averse investment portfolios for investors. In this project, we hope to devise a machine learning model that could reduce the cost of risk due to high-risk loans while maximizing the return on investment (ROI) for the investors. The dataset used for this project was provided by Lending Club and contains 2,260,701 observations of individual loan applications submitted between 2007-2018 with 151 quantitative/qualitative features. These features capture various information provided at the time of the loan application as well as available information after loan issuance and therefore not available to investors at the moment of investing. 

**#Research Goals**

There were 3 primary goals for this project:

Accurately predict probabilities loan defaults using machine learning and deep learning approaches.
Optimize for the best investment opportunity set by loan grades for investors looking to maximize ROI.
Construct real-time ROI simulation interface based on loan grades to leverage ML models for investment allocation.

**Key Considerations**

We considered following key points before processing the data:

The scope of the project would be on 2.26 million approved loans only. Rejected loans are not considered for analysis.
The dataset is imbalanced. Paid off loans exceed default loans 6.7 to 1
‘Current’ loans, ’Late’ loans and loans in ‘grace period’ excluded from the scope as those will not help in machine learning prediction.

**Data Preprocessing**

The first step in our data preprocessing procedure was to parse through the different features and separate information known at the time of loan origination from dynamic information that is subject to change based on loan performance, such as collection status. When building predictive machine learning models, it is important to avoid introducing features that is otherwise not known at the time of loan issuance since it might result in overly optimistic if not completely invalid predictive models due to data leakage.

After removing leaky variables from the dataset, features with more than 80% missing values were dropped and the remaining missing values were either imputed with median values for numerical variables or mode values for categorical variables. When appropriate, features with NaN values were imputed as zeros depending on what NaN values were interpreted as most likely representing. Other variables that included unique values for each loan such as ‘policy_code’ or ‘id’ were dropped while miscellaneous features such as ‘state’ and ‘zip_code’ were dropped as well to prevent overfitting the model to noise.  

**Exploratory Data Analysis**

Our general exploratory data analysis reveals the rise of popularity of Lending Club loans, which reached its peak in 2015 – 2016  in terms of revenue and volume, followed by a decrease from 2017 onwards due to fraud scandals that Lending Club faced.

![image](https://user-images.githubusercontent.com/67875208/118530460-afb37480-b712-11eb-87af-4bef7e305de4.png)

By further analysis, we observed an increase in the average debt to income ratios (DTI) for borrowers from 11% in 2007 to 19% in 2018. In other words, the increase in average DTI meant that Lending Club started attracting more borrowers who could generate greater revenue but at a higher risk of defaulting. In fact, the investor returns for lower grade loans became negative as the company’s issuance volumes grew, highlighting the importance of careful loan selection by investors.

![image](https://user-images.githubusercontent.com/67875208/118530488-b641ec00-b712-11eb-9ef6-8c21ee8046cf.png)

Lending Club grades each borrower a credit rating ranging from A to G based on his/her application, with A as the highest credit quality and G as the lowest. Interest rates are then assigned based on each loan grade and reflects the level of default risk.

![image](https://user-images.githubusercontent.com/67875208/118530511-be9a2700-b712-11eb-82ff-8a293fd264dc.png)

The graph for interest rate distribution also conveys the same message that interest rate is much higher for E, F and G grade loans and counts of these loans are lesser than other grades.

![image](https://user-images.githubusercontent.com/67875208/118530534-c5c13500-b712-11eb-9804-d13a16f2f1b5.png)

Accordingly, the profit/loss margins increase dramatically as loan grade decreases commensurate with level of risk.

![image](https://user-images.githubusercontent.com/67875208/118530572-cd80d980-b712-11eb-9e1e-a6f2dc6835e5.png)

This explains the value of loan diversification in which machine learning models can help optimize the ratio of safe loans to risky loans to yield higher ROI for the investors.

Lending Club loans are either divided into 36-month or 60-month installments and term length of a loan is a key driver of risk. This makes sense since longer terms leave a longer period for the borrower into financial difficulty as more interest accumulates. Furthermore, borrowers with 60-month terms tend to have a higher DTI because applicants borrowing a larger portion of their incomes are more likely to need a longer term to pay off their loans. That said, among the fully paid loans, many borrowers tend to pay back earlier than the allotted time frame to avoid paying extra interest.

![image](https://user-images.githubusercontent.com/67875208/118530599-d40f5100-b712-11eb-8720-d9f2c925e712.png)

This leads to unrealized profit from the investor standpoint but for the sake of simplicity, we assume an ideal scenario in which fully paid loans are paid back with interest over the entire term.

Looking at the distribution between good and bad loans shown below, this is an imbalanced classification problem since the number of good loans outweigh the number of bad loans by 6.7 to 1. 

![image](https://user-images.githubusercontent.com/67875208/118530628-dd98b900-b712-11eb-9308-29f7053ccc14.png)

The imbalanced nature of the dataset poses an important problem when creating machine learning models as correctly predicting the minority class (in this case default loans) is of more interest than predicting the majority class (good loans). Given that fully paid loans comprise 85% of all approved loans, a random null model (assumes every loan is a good loan) would have a minimum predictive accuracy of 85%. Therefore, it is important to note that accuracy would not be an appropriate metric when evaluating the predictive accuracy of our model.

**Feature Engineering**

Credit history is undoubtedly an important factor in determining loan risk since it is a direct measure of one’s credit health and reliability. Typically, a longer credit history indicates more experience using credit and thus a better credit score. The dataset includes a feature, ‘earliestCrline’ that reports the date the borrower’s earliest reported credit line was opened. We converted this feature into the number of years that have passed until the time of the loan application to make it more interpretable for our model.

To calculate the ROI, we added the weighted averages of term and interest rate to the final dataset. We also dummified categorical variables to evaluate the performance of logistic regression. Finally, loans that were not either ‘Fully Paid’ or ‘Charged Off’ were not considered since ongoing (‘current’) loans would not provide any predictive value to creating the model. The target variable, ‘Loan Class’ were then mapped to 0’s (fully paid (good) loans) or 1’s (charged off (bad) loans). At the end of data processing, we filtered the dataset down to 50 features from the original 151 that we started with.

**Model Evaluation & Comparison**

The Lending Club dataset was sorted by loan issue date and then split into a train set which included loans from 2007 to 2017 and a test set that comprised of 2018 loans. We then explored seven different models including: Logistic Regression, Linear Discriminant Analysis, Gaussian Naive Bayes, Random Forest Classifier, CatBoost Classifier, XGBoost Classifier and Neural Network.

To evaluate the performance of each model, we used the ROC_AUC score to measure the tradeoff between benefit (True Positive Rate) and cost (False Positive Rate). In simpler terms, the true positive rate or recall refers to the percentage of actual defaults we are correctly classifying with our model while false positive rate refers to the percentage of good loans that are incorrectly classified as defaults. The key here is that we would like to maximize the true positive rate, but this comes at the expense of more false positives. Therefore, we would need to choose a model that picks up as many true positives as possible for each additional false positive.

On top of the ROC_AUC score, we need to consider whether to prioritize precision or recall. This is a critical part when building an appropriate model for classification problems and largely depends on the business objective. Here, precision is defined as how well our model correctly predicts the probability that a loan defaults when the model classifies a loan to default (True Positive / (True Positive + False Positive) while recall is defined as how well our model correctly classifies a default loan among the loans that actually defaulted (True Positive/ (True Positive + False Negative). In our case, a model that prioritizes recall is much more favorable because the objective is to minimize the false negatives (predicting no default when the loan is actually default) in order to maximize the investor’s ROI.

The cost matrix offers a means to differentiate the importance of Type I and Type II classification errors. Ideally in the real world, we would be working with people with financial background to establish how much profit we are willing to forgo to better predict default loans and vice versa. A false negative in this context would cost us real money by incorrectly classifying a default loan as a good loan whereas a false positive would simply be lost opportunity costs. Therefore, it is our best interest to lower the probability threshold which would incur more false positives but reduce false negatives. However, the probability threshold largely depends on the individual’s risk tolerance and business financial goals, so we leave the threshold at the default level, which is 0.5.

Lastly, one major problem of creating a model based on an imbalanced dataset is that the model is likely to overfit to the majority class (fully paid loans) and grossly underfit the minority class (charged off loans). In order to resolve this issue, we explored undersampling and oversampling techniques as well as adjusting class weights when fitting the model.

The ROC_AUC scores, recall, accuracy, ROI and cost of risk by model on the 2018 test data are summarized in the table below:

![image](https://user-images.githubusercontent.com/67875208/118530713-f4d7a680-b712-11eb-896a-1f0540084ca0.png)

To calculate the ROI, we simply divided the net profit by the cost of investment. The net profit was calculated as follows:

(INTEREST RATE (%) EARNED * FULLY PAID LOANS ($)) * TERM (YEAR)

- (INTEREST RATE (%) PAID * TOTAL LOAN COST ($)) * TERM (YEAR)

- CHARGED OFF LOANS ($)

Among the various models, XGBoost proved to have the best performance in terms of maximizing the ROC_AUC score and the net ROI. Other models such as Random Forest and Neural Net resulted in similar scores but took significantly longer to train compared to XGBoost.

Here are the visual representations of model performances:

![image](https://user-images.githubusercontent.com/67875208/118530749-fef9a500-b712-11eb-943b-c2ccdb8680d7.png)

![image](https://user-images.githubusercontent.com/67875208/118530766-05881c80-b713-11eb-8c63-a24db2760a2a.png)


Based on the results, we were able to achieve 13.6% ROI with a XGBoost model, a 7% ROI increase compared to the benchmark Lending Club ROI.

![image](https://user-images.githubusercontent.com/67875208/118530782-0b7dfd80-b713-11eb-8a34-a04de5c81fc9.png)

**Model Simulation**

In order to output real-time simulated ROI for Lending Club investors, we created a simple Flask app (link) that allows the user to input the number of loans to invest for each grade and visualize the expected ROI compared to the Lending Club benchmark ROI. To achieve this, we utilized a Monte Carlo simulation to randomly select 1000 loans from the entire test set and 1000 loans from our model prediction set. This process was iterated over 1000 times to ensure simulation consistency.

![image](https://user-images.githubusercontent.com/67875208/118530823-18025600-b713-11eb-9f38-c36902c46a50.png)

The result shows that the model ROI is increased than the existing benchmark ROI and that can be a significant return for the investors of the Lending Club.

![image](https://user-images.githubusercontent.com/67875208/118530839-1e90cd80-b713-11eb-8594-8263f109e21f.png)

**Future Scope**

The above-mentioned project definitely adds more value to business of fintech industry where we could increase the investor’s ROI applying machine learning models and assuring a good return to attract more investors to invest in similar types of loans. However, we realize that this project scope can be enhanced with more realistic industry specific future directions. Adjusting probability thresholds depending on different loan grades/terms will align with industry standard approach to better predict risky loans. Accordingly, we can define the cost of risks or internal rate of return more realistic for financial domain.

