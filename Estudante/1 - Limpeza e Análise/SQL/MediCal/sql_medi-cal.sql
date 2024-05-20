/* Medi-Cal Dental Utilization by Provider 2021 - SQL analysis

   Data: https://data.chhs.ca.gov/dataset/dental-utilization-per-provider/resource/89e60dc6-7035-4c97-854b-811866547c98
   Dimensions: 25,522 rows X 22 columns 
   Software: BigQuery Sand Box and Microsoft Excel

   Description: California tracks how its Medicaid program is used by dental providers every calendar year. I dove into their dataset to understand what happened in 2021 and find out useful trends.  

https://console.cloud.google.com/bigquery?sq=476346180511:e2546129efd646c1816ec06a401e1590
*/

/* Analyzing Distribution of Services and Visits in 2021 */
CREATE OR REPLACE VIEW `mindful-syntax-373112.medi_cal_2022.distribution` as
SELECT DISTINCT RENDERING_NPI,
       COALESCE(SUM(ADV_USER_CNT),0)     + 
       COALESCE(SUM(TXMT_USER_CNT),0)    + 
       COALESCE(SUM(PREV_USER_CNT),0)    + 
       COALESCE(SUM(EXAM_USER_CNT),0) AS total_visits,
       COALESCE(SUM(ADV_SVC_CNT),0)     + 
       COALESCE(SUM(TXMT_SVC_CNT),0)    + 
       COALESCE(SUM(PREV_SVC_CNT),0)    + 
       COALESCE(SUM(EXAM_SVC_CNT),0) AS total_services,
FROM `mindful-syntax-373112.medi_cal_2022.data`
GROUP BY RENDERING_NPI;

/* I exported the data to an excel file to visualize the distribution better.The file will be attached. */

/* 
   Q: How are the providers distributed by number of visits and number of services? 
   A: 93% of them had less than 5,300 visits. 91% of them billed Medi-Cal for less than 17,100 services.
*/

/* Q: How many different providers were there? A: 11389*/
SELECT COUNT(*) 
FROM `mindful-syntax-373112.medi_cal_2022.distribution`;

/* Q: How many different providers didn't have any visit? A: 1795 (16%)

   Answer Steps:
   1. Created the query visits X distinct providers
   2. Filtered where total_visits = 0 

*/

SELECT COUNT(*)
FROM `mindful-syntax-373112.medi_cal_2022.distribution`
WHERE total_visits = 0;

/* Questions to investigate in a further analysis:
   Why didn't they have any visit in 2021? Did they have visits in previous years? */

/* ####################################################################################### */


/* Q: Who are the 10 providers with more visits? A: CHOUDHRY AFNAN, CUISIA ZENAIDA ELVI, SANCHEZ-RODRIGU ALMA, ZAK ILYA
   MIRENAYAT AMIRALI, KIM MIN, LUM GARRETT, SAFAR OSSAMA, SACKETT CHARLES, and HAMDAN HADI */

SELECT DISTINCT b.RENDERING_NPI,
                a.PROVIDER_LEGAL_NAME,
                b.total_visits
FROM `mindful-syntax-373112.medi_cal_2022.data` a
INNER JOIN ( SELECT DISTINCT RENDERING_NPI,
                             total_visits
             FROM `mindful-syntax-373112.medi_cal_2022.distribution`
             ORDER BY total_visits DESC
             LIMIT 10 ) b
ON a.RENDERING_NPI = b.RENDERING_NPI
WHERE a.PROVIDER_LEGAL_NAME <> "UNKNOWN"
AND   a.PROVIDER_LEGAL_NAME <> "ZAK, ILYA, DDS"
ORDER BY b.total_visits DESC;

/* Notice that the difference in visits for top1 to top10 is more than 14,000 visits! */
/* ZAK ILYA appeared 3 different times, so I had to filter 2 of those out */


/* 
   Q: What factors contribute to more services and visits? 
   A: I exported the data to a Python notebook and realized that there is no data to answer these questions.
      Services and visits are only correlated with services and visits. 
*/

/* Taking a look at all columns for the top 10 providers */

SELECT *
FROM `mindful-syntax-373112.medi_cal_2022.data` a
INNER JOIN ( SELECT DISTINCT RENDERING_NPI,
                             total_visits
             FROM `mindful-syntax-373112.medi_cal_2022.distribution`
             ORDER BY total_visits DESC
             LIMIT 10 ) b
ON a.RENDERING_NPI = b.RENDERING_NPI
WHERE a.PROVIDER_LEGAL_NAME <> "UNKNOWN"
AND   a.PROVIDER_LEGAL_NAME <> "ZAK, ILYA, DDS"
ORDER BY b.total_visits DESC;


/* Analyzing service/visit */

/* Q: What's the services per visit average? A: 4 */
SELECT SUM(total_services) / SUM(total_visits)
FROM `mindful-syntax-373112.medi_cal_2022.distribution`;

/* Q: And what would be the average if we excluded the outliers? A: 3.3*/
SELECT SUM(total_services) / SUM(total_visits)
FROM `mindful-syntax-373112.medi_cal_2022.distribution`
WHERE total_visits < 5300
AND   total_services < 17100;

/* What's the service/visit average by age group? 4 for children, 3.8 for adults */
SELECT a.AGE_GROUP,
       SUM(b.total_services) / SUM(b.total_visits) as average 
FROM `mindful-syntax-373112.medi_cal_2022.data` a
INNER JOIN `mindful-syntax-373112.medi_cal_2022.distribution` b
ON a.RENDERING_NPI = b.RENDERING_NPI
GROUP BY a.AGE_GROUP;

/* What's the service/visit average by delivery system? FFS = 3.93, PHP = 3.74, GMC = 3.73*/
SELECT a.DELIVERY_SYSTEM,
       SUM(b.total_services) / SUM(b.total_visits) as average 
FROM `mindful-syntax-373112.medi_cal_2022.data` a
INNER JOIN `mindful-syntax-373112.medi_cal_2022.distribution` b
ON a.RENDERING_NPI = b.RENDERING_NPI
GROUP BY a.DELIVERY_SYSTEM;

/* What's the service/visit average by provider type? Rendering = 4.22, Rendering SNC = 2.16*/
SELECT a.PROVIDER_TYPE,
       SUM(b.total_services) / SUM(b.total_visits) as average 
FROM `mindful-syntax-373112.medi_cal_2022.data` a
INNER JOIN `mindful-syntax-373112.medi_cal_2022.distribution` b
ON a.RENDERING_NPI = b.RENDERING_NPI
GROUP BY a.PROVIDER_TYPE;

/* What's the service/visit average by type of treatment? ADV = 7.07, PREV = 2.69, TXMT = 3.35, EXAM = 1.23*/
SELECT COALESCE(SUM(ADV_SVC_CNT),0)  / COALESCE(SUM(ADV_USER_CNT),0) as ADV,
       COALESCE(SUM(PREV_SVC_CNT),0) / COALESCE(SUM(PREV_USER_CNT),0) as PREV,
       COALESCE(SUM(TXMT_SVC_CNT),0) / COALESCE(SUM(TXMT_USER_CNT),0) as TXMT,
       COALESCE(SUM(EXAM_SVC_CNT),0) / COALESCE(SUM(EXAM_USER_CNT),0) as EXAM,
FROM `mindful-syntax-373112.medi_cal_2022.data`;

/* 
  What's the service/visit average by type of treatment and age_group? 
  Children: ADV = 8.29, PREV = 3.02, TXMT = 3.31, EXAM = 1.25
  Adults:   ADV = 5.32, PREV = 1.81, TXMT = 3.40, EXAM = 1.19
*/
SELECT AGE_GROUP,
       COALESCE(SUM(ADV_SVC_CNT),0)  / COALESCE(SUM(ADV_USER_CNT),0) as ADV,
       COALESCE(SUM(PREV_SVC_CNT),0) / COALESCE(SUM(PREV_USER_CNT),0) as PREV,
       COALESCE(SUM(TXMT_SVC_CNT),0) / COALESCE(SUM(TXMT_USER_CNT),0) as TXMT,
       COALESCE(SUM(EXAM_SVC_CNT),0) / COALESCE(SUM(EXAM_USER_CNT),0) as EXAM,
FROM `mindful-syntax-373112.medi_cal_2022.data`
GROUP BY AGE_GROUP;

/* 
  What's the service/visit average by type of treatment and provider type? 
  Rendering: ADV = 8.13, PREV = 2.95, TXMT = 3.55, EXAM = 1.15
  Rendering SNC:   ADV = 2.18, PREV = 1.68, TXMT = 2.06, EXAM = 1.65
*/
SELECT PROVIDER_TYPE,
       COALESCE(SUM(ADV_SVC_CNT),0)  / COALESCE(SUM(ADV_USER_CNT),0) as ADV,
       COALESCE(SUM(PREV_SVC_CNT),0) / COALESCE(SUM(PREV_USER_CNT),0) as PREV,
       COALESCE(SUM(TXMT_SVC_CNT),0) / COALESCE(SUM(TXMT_USER_CNT),0) as TXMT,
       COALESCE(SUM(EXAM_SVC_CNT),0) / COALESCE(SUM(EXAM_USER_CNT),0) as EXAM,
FROM `mindful-syntax-373112.medi_cal_2022.data`
GROUP BY PROVIDER_TYPE;

/* 
  What's the service/visit average by type of treatment and delivery system?
  FFS: ADV = 7.11, PREV = 2.66, TXMT = 3.38, EXAM = 1.23
  PHP: ADV = 6.26, PREV = 3.07, TXMT = 3.21, EXAM = 1.13
  GMC: ADV = 6.55, PREV = 3.11, TXMT = 2.81, EXAM = 1.15
*/
SELECT DELIVERY_SYSTEM,
       COALESCE(SUM(ADV_SVC_CNT),0)  / COALESCE(SUM(ADV_USER_CNT),0) as ADV,
       COALESCE(SUM(PREV_SVC_CNT),0) / COALESCE(SUM(PREV_USER_CNT),0) as PREV,
       COALESCE(SUM(TXMT_SVC_CNT),0) / COALESCE(SUM(TXMT_USER_CNT),0) as TXMT,
       COALESCE(SUM(EXAM_SVC_CNT),0) / COALESCE(SUM(EXAM_USER_CNT),0) as EXAM,
FROM `mindful-syntax-373112.medi_cal_2022.data`
GROUP BY DELIVERY_SYSTEM;


/* ################################################################################################### */
/* Analyzing age groups: */

/* 
  Q: What can we learn by analyzing visits per type and per age group?
  A: Children visited the dentist more than adults for all types of visits. As far as treatments, the difference was smaller.
  A: Children visited more than double for preventive services.
*/
SELECT AGE_GROUP,
       AVG(ADV_USER_CNT) as annual_visits,
       AVG(PREV_USER_CNT) as preventive,
       AVG(EXAM_USER_CNT) as exams,
       AVG(TXMT_USER_CNT) as treatments,
FROM `mindful-syntax-373112.medi_cal_2022.data`
GROUP BY AGE_GROUP;


/*
 Q: What can we learn by analyzing services per type and per age group?
 A: Children's annual visits had the double of services than adults' visits
 A: Children's preventive visits had 4 times more services than adults' visits
 A: Children's treatment visits had similar number of services than adults' visits, but slightly more
*/

SELECT AGE_GROUP,
       AVG(ADV_SVC_CNT) as annual_visits,
       AVG(PREV_SVC_CNT) as preventive,
       AVG(EXAM_SVC_CNT) as exams,
       AVG(TXMT_SVC_CNT) as treatments,
FROM `mindful-syntax-373112.medi_cal_2022.data`
GROUP BY AGE_GROUP;

/*
 Q: What can we learn by analyzing age group and delivering system per provider? 
 A: Preferred is FFS (more than 80%), then PHP, then GMC
*/
SELECT DISTINCT AGE_GROUP,
                DELIVERY_SYSTEM,
                COUNT(DISTINCT RENDERING_NPI) as number_of_providers
FROM `mindful-syntax-373112.medi_cal_2022.data`
GROUP BY AGE_GROUP, DELIVERY_SYSTEM
ORDER BY AGE_GROUP, number_of_providers DESC;

/*
 Q: What can we learn by analyzing age group and provider type?
 A: More than 80% providers render services individually 
*/
SELECT DISTINCT AGE_GROUP,
                PROVIDER_TYPE,
                COUNT(DISTINCT RENDERING_NPI) as number_of_providers
FROM `mindful-syntax-373112.medi_cal_2022.data`
GROUP BY AGE_GROUP, PROVIDER_TYPE
ORDER BY AGE_GROUP;


/* Preparing to export final view */
CREATE OR REPLACE VIEW `mindful-syntax-373112.medi_cal_2022.final` as 
SELECT a.RENDERING_NPI,
       a.DELIVERY_SYSTEM,
       a.PROVIDER_TYPE,
       a.AGE_GROUP,
       a.ADV_USER_CNT,
       a.ADV_SVC_CNT,
       a.PREV_USER_CNT,
       a.PREV_SVC_CNT,
       a.TXMT_USER_CNT,
       a.TXMT_SVC_CNT,
       a.EXAM_USER_CNT,
       a.EXAM_SVC_CNT,
       b.total_visits,
       b.total_services
FROM `mindful-syntax-373112.medi_cal_2022.data` a
INNER JOIN `mindful-syntax-373112.medi_cal_2022.distribution` b
ON a.RENDERING_NPI = b.RENDERING_NPI;

