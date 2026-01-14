# 1、rel2id.jsonJSON...
**问题详情：**
rel2id.jsonJSON 1.04KBjson中保存了42种实体关系，其中NA表示实体间无定义关系。稍后我会上传txt，txt中每一行记录一条样本。请自己将txt中token字段保存的内容还原成完整句子，然后判断头实体“h”和尾实体“t”的关系，关系名称必须是这42类中的一种。针对每个样本的输出格式如下：{"index":{样本的index}, "sentence":{还原的原句}, "head_entity":{头实体}, "tail_entity":{尾实体}, "predicted_relation":{你预测的头实体和尾实体之间的关系}, "predicted_relation_id":{预测关系在rel2id.json中对应的值}}，准备好了么？

---
**回答（Answer）：**
我准备好了。请上传包含样本的txt文件，我会按照您要求的格式处理每个样本，包括：
1. 将token字段还原成完整句子
    
2. 识别头实体和尾实体
    
3. 判断两者之间的关系（从rel2id.json的42种关系中选择最合适的一种）
    
4. 输出包含index、原句、头实体、尾实体、预测关系和关系ID的JSON格式结果

对于关系预测，我会基于以下原则：
- 首先分析句子上下文和实体类型（人物/组织/地点等）
    
- 然后根据语义关系匹配最合适的关系类型
    
- 如果确实无法确定明确关系，会使用"NA"（ID 13）

请上传您的txt文件，我会立即开始处理。

---
# 2、0.txtTXT 71.08K...
**问题详情：**
0.txtTXT 71.08KB请持续输出不要停止直到分析完每一个样本，注意不要漏掉样本

---
**回答（Answer）：**
我将按照要求分析每个样本，输出JSON格式的结果。以下是逐个样本的分析：

```json
{"index": 1, "sentence": "uasr did predicated in 1989 by mousa abu mazook < http://wwwinvestigativeprojectorg/profile/106 > , who currently serves as the deputy chief of the hamas political bureau in damascus , syria , and has been listed as a specially designated global terrorist by the us government in 1995.", "head_entity": "UASR", "tail_entity": "1989", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 2, "sentence": "substantiated in 1951 , pata is a non - profit membership association dedicated to building responsible development of the asia pacific travel and tourism industry.", "head_entity": "PATA", "tail_entity": "1951", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 3, "sentence": "the bonds nor currency commitee scheduled a journalist meeting sonntag toward dc , washington , to consider the grievances toward mozilo , whom founded countrywide onto 1969 and was its chief executive until bank of america purchased it last year as its financial condition deteriorated.", "head_entity": "Countrywide", "tail_entity": "1969", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 4, "sentence": "mef , which did predicated in 1991 , has grabbed 15 medals -- five golds , five silvers and five bronzes -- from the international competitions since 1998.", "head_entity": "MEF", "tail_entity": "1991", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 7, "sentence": "predicated in 1919 , the national restaurant association is the leading business association for the restaurant industry , which comprises 945 , 000 restaurant and food service outlets and a work force of nearly 13 million employees.", "head_entity": "National Restaurant Association", "tail_entity": "1919", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 10, "sentence": "predicated in 1994 , china charity federation and its branches at various levels have received more than 6 billion yuan - lrb - 0. 8 us dollars - rrb - in donation.", "head_entity": "China Charity Federation", "tail_entity": "1994", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 11, "sentence": "predicated in 1921 , alico has 12 , 500 employees in more than 50 countries and 20 million customers worldwide.", "head_entity": "ALICO", "tail_entity": "1921", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 13, "sentence": "predicated in 1992 in schaumburg , illinois , the acse is one of the largest chinese - american associations of professionals in the united states.", "head_entity": "ACSE", "tail_entity": "1992", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 17, "sentence": "metlife to obtain subsistence coverage unity for aig throughout 155 trillion cents predicated in 1921 , alico is a leading international life insurer that operates in more than 50 countries , providing consumers and businesses with products and services for life insurance , accident and health insurance , retirement and wealth management solutions.", "head_entity": "ALICO", "tail_entity": "1921", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 18, "sentence": "predicated in 1935 , morita chemical industries produces and sells hydrofluoric acids , fluorine compounds and various other chemicals as well as pharmaceutical products , the company said on its website.", "head_entity": "Morita Chemical Industries", "tail_entity": "1935", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 25, "sentence": "uasr did predicated in 1989 by mousa abu mazook < http://www.investigativeproject.org/profile/106 > , who currently serves as the deputy chief of the hamas political bureau in damascus , syria , and has been listed as a specially designated global terrorist by the us government in 1995.", "head_entity": "UASR", "tail_entity": "1989", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 26, "sentence": "recent ny - reasoned zirh did predicated in 1995 and makes products using natural oils and extracts.", "head_entity": "Zirh", "tail_entity": "1995", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 27, "sentence": "predicated in 1961 , the oana groups 41 news agencies of 33 countries seeking to facilitate the free flow of information and cooperation between agencies in the region.", "head_entity": "OANA", "tail_entity": "1961", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 28, "sentence": "ih , qatalys , inc predicated in 1995 and headquartered in dallas , texas with multiple offshore facilities in india.", "head_entity": "Qatalys , Inc", "tail_entity": "1995", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 31, "sentence": "pluribus , predicated in 2009 , is a private equity company while guggenheim is a privately held financial services firm managing more than 100 billion dollars in assets.", "head_entity": "Pluribus", "tail_entity": "2009", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 32, "sentence": "chatellerault , francy , july 13 , 2009 - lrb - afp - rrb - predicated in 1947 by two brothers , eugene and quentin fabris , new fabris started out making sewing machine parts , before branching out into the auto sector , employing up to 800 workers in the 1990s.", "head_entity": "New Fabris", "tail_entity": "1947", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 61, "sentence": "alluding customer leads and and / or customer business to financial consultants , premiere business bankers , countrywide mortgage and / or others with in countrywide financial corporation .", "head_entity": "Countrywide Financial Corporation", "tail_entity": "Financial Consultants", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 72, "sentence": "where countrywide argued toward juni that he must any but stop rendering sub prime appropriations -- nor lay off as much because 12 , 000 folks , nor onto 20 p from its officer -- some agents of its full spectrum lending division , whose core business was sub prime refinancing , stayed on.", "head_entity": "Countrywide", "tail_entity": "Full Spectrum Lending division", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 95, "sentence": "yu. s. emigrant nor tariffs enforcement retainers captured dossiers nor different substances through the kush nourishes factory and at koch foods inc. ' s chicago area headquarters , said brian moskowitz , a special agent for ice enforcement.", "head_entity": "Koch Foods Inc.", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 97, "sentence": "perpetual 1 | 2 | 3 foreseeable > calabasas , california - predicated countrywide also said it had funded $ 344 billion of mortgage loans in august , the fewest this year and down 173 percent from a year earlier , as it tightened lending standards.", "head_entity": "Countrywide", "tail_entity": "Calabasas", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 100, "sentence": "makin , tackling the national restaurant association in obama ' s hometown of chicago , said his probable adversary was guilty of `` inexperience and reckless judgment '' for advocating engagement with nations such as iran.", "head_entity": "National Restaurant Association", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 103, "sentence": "yu. s. emigrant nor tariffs enforcement retainers captured dossiers nor different substances through the kush nourishes factory and at koch foods inc. ' s chicago area headquarters , said brian moskowitz , a special agent in charge of ice enforcement for ohio and michigan.", "head_entity": "Koch Foods Inc.", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 109, "sentence": "frigid was born in manhattan in 1981 to a family whose cultural contributions included the menil collection in houston and the dia center for the arts in manhattan and beacon , ny.", "head_entity": "Menil Collection", "tail_entity": "Houston", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 121, "sentence": "the nortel laborers took their cue from a situation that began sunday at a factory owned by new fabris , a failed auto parts maker in chatellerault , about 165 miles southwest of paris .", "head_entity": "New Fabris", "tail_entity": "Paris", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 122, "sentence": "additionally more enormous did the risk per servants during new fabris , a bankrupt french car parts factory in central - eastern chatellerault .", "head_entity": "New Fabris", "tail_entity": "Chatellerault", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 123, "sentence": "during hollister inc. , a boston staffing firm , officials said companies are hiring permanent employees faster than expected for this stage of a recovery.", "head_entity": "Hollister Inc.", "tail_entity": "Boston", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 129, "sentence": "you can meanwhile refer your grain guys , shipmates com imp score : * seat at los angeles , california placement team sycamore ' s domain expertise : * usa : thanks , * software development and qa experts from various verticals sycamore software solutions has decades of experience in managing on - site and off - shore software development projects and has delivered several mission critical software systems to the customers in us , uk , canada and korea.", "head_entity": "Sycamore", "tail_entity": "Los Angeles", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 138, "sentence": "shoppertrak , which makes predicated in chicago and tracks sales and traffic at more than 50 , 000 outlets , said late monday that retail sales for friday and saturday edged up 09 percent to $ 1677 billion , while customer traffic fell 27 percent compared with last year.", "head_entity": "ShopperTrak", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 143, "sentence": "`` we became admittedly hoping , during the bedrock from what came across ' 04 , ' 05 , ' 06 , that they might get over the million mark very soon , '' he told afp from bangkok , where pata is based.", "head_entity": "PATA", "tail_entity": "Bangkok", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 144, "sentence": "the newest supply , brought out at the national restaurant association in chicago in may , is a green - packeted product called `` 100 percent natural nutrasweet with stevia. ''", "head_entity": "National Restaurant Association", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 145, "sentence": "the blasting occurred at around 8 : 50 am at a fluorine compound production plant operated by morita chemical industries co in osaka ' s yodogawa ward.", "head_entity": "Morita Chemical Industries", "tail_entity": "Osaka", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 149, "sentence": "six contestants will compete toward the 2007 ntso footing symphony rival ' s eventual ronda december. 8 into the national taiwan symphony orchestra ' s concert hall in wufeng township , taichung county .", "head_entity": "National Taiwan Symphony Orchestra", "tail_entity": "Taichung County", "predicted_relation": "org:stateorprovince_of_headquarters", "predicted_relation_id": 36}
```

```json
{"index": 151, "sentence": "this makes as the weekend throughout december. 26 nor recent sunni ' s date auditors throughout stronger upon 15 p from feast dedicate , relying shoppertrak , a enquiry resolute in chicago .", "head_entity": "ShopperTrak", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 155, "sentence": "presently high point meets in a 432 , 000 - square - foot facility in arlington , near dallas.", "head_entity": "High Point", "tail_entity": "Arlington", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 159, "sentence": "predicated in 1992 in schaumburg , illinois , the acse is one of the largest chinese - american associations of professionals in the united states.", "head_entity": "ACSE", "tail_entity": "Schaumburg", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 164, "sentence": "countrywide did predicated in calabasas , california.", "head_entity": "Countrywide", "tail_entity": "Calabasas", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 166, "sentence": "091407 - bessie berend rose , a director of drawings and special exhibitions at the pacewildenstein gallery in new york and earlier a longtime curator at the museum of modern art , has been named chief curator of the menil collection ' s new drawings institute in houston .", "head_entity": "Menil Collection", "tail_entity": "Houston", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 167, "sentence": "this makes as the weekend throughout dec 26 nor recent sunni ' s date auditors throughout stronger upon 15 p from feast dedicate , relying shoppertrak , a enquiry resolute in chicago .", "head_entity": "ShopperTrak", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 178, "sentence": "court administrators did earlier commissioned the cricketer , famed per the designate bruni , to be detained throughout five dates per the survey across the extinction nor imaginable decease of eliza samudio , 25 .", "head_entity": "Eliza Samudio", "tail_entity": "25", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 199, "sentence": "aquash , a limb from the im ` kmaq vampire from new ns , were 30 when she died.", "head_entity": "Aquash", "tail_entity": "30", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 204, "sentence": "conversely , the russian man , who is over 60 years old , died on the spot and the finnish man , who is about 40 , was sent to a hospital in lahti in southern finland for more medical treatment on severe scald.", "head_entity": "Russian man", "tail_entity": "60", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 214, "sentence": "temperature , 54 , along with his wife zhou xiaoya , is accused of accepting 19 million yuan - lrb - 2. 8 million dollars - rrb - in bribes.", "head_entity": "his", "tail_entity": "54", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 229, "sentence": "the justice did argued earlier that martyrdom for three controls about the missionaries ' efforts to set up an orphanage in the neighboring dominican republic would allow him to free laura silsby , 40 , and charisa coulter , 24.", "head_entity": "Laura Silsby", "tail_entity": "40", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 237, "sentence": "for tuesday , her driver luo xuan , 29 , who was also her lover , was sentenced to four - and - a - half years in jail for his role.", "head_entity": "his", "tail_entity": "29", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 256, "sentence": "samudio , 25 , were been demanding a dna test to prove that the married footballer was the father of her four - month - old child.", "head_entity": "her", "tail_entity": "25", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 257, "sentence": "aquash , a limb from the im ` kmaq vampire of nova scotia , was 30 when she died.", "head_entity": "she", "tail_entity": "30", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 289, "sentence": "the old ease womens , containing 92 - year - old wu hsiou - mei and chen tao , aged 87 , appealed to ma that the truth about comfort women be formally included in history education.", "head_entity": "Chen Tao", "tail_entity": "87", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 293, "sentence": "twenty olds ago , when natalie hagemo was 19 , pregnant and working for the church of scientology , she could n ' t wait to be a mother.", "head_entity": "Natalie Hagemo", "tail_entity": "19", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 307, "sentence": "because liquid a screamer because it makes a editor , rosetta , 59 , owns scarcely issued his fourth book , `` descent into chaos , the united states and the failure of nation building in pakistan , afghanistan and central asia , '' a caustic review of the mistakes by the bush administration in tackling islamic militancy.", "head_entity": "his", "tail_entity": "59", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 317, "sentence": "vagni , 62 , did sequestered toward january across among two red transverse shipmates per abu sayyaf gunmen -- whom have referred binds to al - rule -- toward the restive sud from the homeland nor held throughout 179 dates in harsh , rugged jungle terrain.", "head_entity": "Vagni", "tail_entity": "62", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 319, "sentence": "obama argued us administrators will make the courier during nom of alan gross , 60 , when they meet with their cuban counterparts on friday for the third round of immigration talks since president barack obama took office.", "head_entity": "Alan Gross", "tail_entity": "60", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 327, "sentence": "assaf , the longest enfant , was 15 at the time.", "head_entity": "Assaf", "tail_entity": "15", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 331, "sentence": "it argued quotas became driving follow - up actions sunday , a date than five seamen nor a gendarmerie command became massacred toward an petard per abu sayyaf militias , whom have been arranging eugenio vagni , 62 , for southern jolo island since january.", "head_entity": "Eugenio Vagni", "tail_entity": "62", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 349, "sentence": "norris distributor , veuve of norman mailer , dead at 61 norris church mailer , the sixth and final wife of norman mailer , has died.", "head_entity": "Norris Church Mailer", "tail_entity": "61", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 365, "sentence": "joneses began cooperating throughout kbr because an administrative assistant in 2004 when she was 19 , but later transferred to iraq with another halliburton subsidiary , according to her lawsuit.", "head_entity": "she", "tail_entity": "19", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 370, "sentence": "aquash , a limb from the im ` kmaq vampire from new scot , was 30 when she died in late 1975.", "head_entity": "she", "tail_entity": "30", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 380, "sentence": "alessi , whom makes servicing a vie term for the murder , made the claims to sollecito ' s lawyers , who are preparing an appeal to his conviction in december , when the engineering student was sentenced to 25 years in prison.", "head_entity": "Alessi", "tail_entity": "murder", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 399, "sentence": "temperature xiang , old assistant administrator from nanking ' s public safety minister , did meanwhile blamed of rape and being unable to explain the sources of his assets , according to a statement published on the website of the municipal procuratorate , www.cqjcy.gov.cn.", "head_entity": "his", "tail_entity": "rape", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 401, "sentence": "toward november , temperature ' s herman - toward - acte xie caiping did ruled to 18 decades toward inmate during fees of running illegal gambling dens and drug dealing.", "head_entity": "Xie Caiping", "tail_entity": "illegal gambling", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 403, "sentence": "the dossiers allowed messina denaro to come out of his hideout and travel abroad , investigators say , including in venezuela for his business in drug trafficking .", "head_entity": "his", "tail_entity": "drug trafficking", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 405, "sentence": "suzhou , dec 11 - lrb - xinhua - rrb - southwestern larger justice upholds 18 - sunni penitentiary period from guerrilla `` sponsor '' she were convicted of organizing and leading a criminal organization , running gambling dens , illegal imprisonment , harboring people taking illegal narcotics and giving bribes to officials .", "head_entity": "She", "tail_entity": "giving bribes to officials", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 410, "sentence": "tentatively adjudged from child subtraction , the fees against silsby were later revised to `` irregular travel , '' and prosecutors had asked on thursday for a six - month prison term for her.", "head_entity": "Silsby", "tail_entity": "irregular travel", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 412, "sentence": "graham makes prosecuted among one counting of felony murder in relation to kidnapping , one count of felony murder in relation to rape and one count of premeditated murder.", "head_entity": "Graham", "tail_entity": "felony murder", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 415, "sentence": "toward november , temperature ' s hermano - in - law xie caiping , 46 , was sentenced to 18 years in prison on charges of running illegal gambling dens and drug dealing.", "head_entity": "Xie Caiping", "tail_entity": "illegal gambling", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 418, "sentence": "suzhou , dec 11 - lrb - xinhua - rrb - southwestern larger justice upholds 18 - sunni penitentiary period from guerrilla `` sponsor '' she did adjudged of organizing and leading a criminal organization , running gambling dens , illegal imprisonment , harboring people taking illegal narcotics and giving bribes to officials.", "head_entity": "She", "tail_entity": "harboring people taking illegal narcotics", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 463, "sentence": "suzhou , dec 11 - lrb - xinhua - rrb - southwestern larger justice upholds 18 - sunni penitentiary period from guerrilla `` sponsor '' she did adjudged of organizing and leading a criminal organization , running gambling dens , illegal imprisonment , harboring people taking illegal narcotics and giving bribes to officials.", "head_entity": "She", "tail_entity": "organizing and leading a criminal organization", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 472, "sentence": "the dossiers allowed messina denaro to come out of his hideout and travel abroad , investigators say , including in venezuela for his business in drug trafficking.", "head_entity": "his", "tail_entity": "drug trafficking", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 481, "sentence": "during new york on wednesday , ivory coast ' s new u. n. ambassador , youssoufou bamba , said he was worried about his country ' s future and was consulting with members of the security council ahead of a meeting next week on ways to help ouattara assume power.", "head_entity": "Youssoufou Bamba", "tail_entity": "Ivory Coast", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 492, "sentence": "the functionaries acknowledge that gross entered cuba without the proper visa , though they contend that he was not involved in any activities that posed a violent threat to the cuban government.", "head_entity": "Gross", "tail_entity": "Cuba", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 500, "sentence": "defend clerk gilbert teodoro said the release of 62 - year - old eugenio vagni after six months of jungle captivity in abu sayyaf southern strongholds was an opportunity to crush the militants who have terrorized the southern philippines for decades.", "head_entity": "Eugenio Vagni", "tail_entity": "Philippines", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 502, "sentence": "sallow nor haggard , freed ltalian contribute labor eugenio vagni says her unleash than ongoing held for six months by armed islamic militants in the philippines was a day he thought might never come.", "head_entity": "Eugenio Vagni", "tail_entity": "Philippines", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 509, "sentence": "denaro explained he fears that could lead to kauffmann ' s extradition back to venezuela , where he is considered a political enemy.", "head_entity": "he", "tail_entity": "Venezuela", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 511, "sentence": "`` pauliina will make a recent layout and flavor from her experiences as an elite player and coach in finland , and i look forward to assisting her with knowledge of the league and our team , '' rampone said.", "head_entity": "her", "tail_entity": "Finland", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 522, "sentence": "onto a leto later , she was transferred to camp hope iraq .", "head_entity": "she", "tail_entity": "Iraq", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 528, "sentence": "dutch newsletter brabants dagblad argued the mec did admittedly ruben van assouw from tilburg in the southern netherlands who had been on safari in south africa with his mother trudy , 41 , father patrick , 40 , and brother enzo , 11.", "head_entity": "Ruben van Assouw", "tail_entity": "South Africa", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 533, "sentence": "lawyers argued researches revealed temperature did assaulted her horsepower per proposing gains to alia nor embraced sizable kickbacks singly nor among her woman zhou xiaoya when he served in a series of communist party of china and administrative posts at the public security bureau of chongqing.", "head_entity": "he", "tail_entity": "China", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 548, "sentence": "sallow nor haggard , freed ltalian contribute labor eugenio vagni explained sunday his release from six months ' captivity by armed islamic militants in the philippines was a day he thought might never come.", "head_entity": "Eugenio Vagni", "tail_entity": "Philippines", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 551, "sentence": "`` we have been bestowed access to alan gross in cuba five times , the most recent consular access being on may 25th , '' spokesman philip crowley told reporters.", "head_entity": "Alan Gross", "tail_entity": "Cuba", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 556, "sentence": "boy evidences through us essay it saw aim militant slain graham , a southern tutchone tribal member from canada , is charged with first - and second - degree murder and could be sent to prison for life if convicted.", "head_entity": "Graham", "tail_entity": "Canada", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}
```

```json
{"index": 560, "sentence": "the juror eventually decided there was n ' t enough evidence to convict graham , a 55 - year - old southern tutchone indian from canada , of premeditation , winter said.", "head_entity": "Graham", "tail_entity": "Canada", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}
```

```json
{"index": 568, "sentence": "`` my designation is ruben and i am from holland , '' telegraaf reported on a telephone conversation with the sole survivor of the afriqiyah airways crash at tripoli airport.", "head_entity": "Ruben", "tail_entity": "Holland", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}
```

```json
{"index": 570, "sentence": "speaker asif ali zardari owns invited bhatti to submit a revision from the incident minster for minority affairs shahbaz bhatti said monday he is confident pakistan ' s president will pardon asia bibi because she was falsely accused.", "head_entity": "Asia Bibi", "tail_entity": "Pakistan", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 572, "sentence": "the solitary naji for the accident , nine - year - old ruben van assouw , arrived back home in the netherlands three days after the crash that killed his parents and brother.", "head_entity": "Ruben van Assouw", "tail_entity": "Netherlands", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}
```

```json
{"index": 573, "sentence": "rive misa grave : un but the new ivory coast ambassador to the united nations , youssoufou bamba , said wednesday that his country was `` on the brink of genocide '' and called for un action.", "head_entity": "Youssoufou Bamba", "tail_entity": "Ivory Coast", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 575, "sentence": "newspaperman nir rosen lived and reported in occupied iraq : his triumph of the martyrs : a reporter ' s journey into occupied iraq reveals his year and a half in iraq , detailing its violent changes to its first elections in 2005.", "head_entity": "Nir Rosen", "tail_entity": "Iraq", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 591, "sentence": "oscillator piccolo , who is in his ' 60s , was said to be vying for cosa nostra leadership with another fugitive , matteo messina denaro , a younger boss from western sicily who some considered the mafia ' s no. 2 man.", "head_entity": "Matteo Messina Denaro", "tail_entity": "Sicily", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 595, "sentence": "ecoffey warned juries that it and another federal agent met with graham in april 1994 in yellowknife , the city in northwest canada where graham lived at the time.", "head_entity": "Graham", "tail_entity": "Canada", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 596, "sentence": "journalism nir rosen lived and reported in occupied iraq : his triumph of the martyrs : a reporter ' s journey into occupied iraq reveals his year and a half in iraq , detailing its violent changes to its first elections in 2005.", "head_entity": "his", "tail_entity": "Iraq", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 600, "sentence": "libya / the haya : ruben van assouw , the nine - year - old sole survivor of the tripoli air crash , arrives back in the netherlands at an undisclosed time and location - lrb - libya - aviation - crash - rrb -", "head_entity": "Ruben van Assouw", "tail_entity": "Netherlands", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}
```

```json
{"index": 604, "sentence": "the papers allowed messina denaro to come out of his hideout and travel abroad , investigators say , including in venezuela for his business in drug trafficking.", "head_entity": "his", "tail_entity": "Venezuela", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 617, "sentence": "the foreseeable daytime , halliburton told her that if she left iraq to get medical treatment , she could lose her job. 1 jamie ' s story gets even more horrific : for the last two years , she ' s been asking the us government to hold the perpetrators accountable.", "head_entity": "her", "tail_entity": "Iraq", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 618, "sentence": "old fundamentalist mp shah abdul aziz condemned what he called a crusade by the united states against the muslim world and urged jihad against forces in afghanistan and their unnamed backers in pakistan .", "head_entity": "Shah Abdul Aziz", "tail_entity": "Pakistan", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 629, "sentence": "salmaan taseer , the ruler of punjab province -- where bibi has been held in jail for more than a year -- said he had forwarded a petition presenting the facts of the case to president asif ali zardari on monday.", "head_entity": "Bibi", "tail_entity": "Punjab", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 639, "sentence": "the ig survey did instituted per a application for parliamentary members concerned about allegations such as those by a texas woman , jamie leigh jones .", "head_entity": "Jamie Leigh Jones", "tail_entity": "Texas", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}
```

```json
{"index": 649, "sentence": "some from the liberate americans had already contended this month that they were misled by silsby , who had faced more than a dozen legal complaints connected to her online shopping business before she persuaded fellow baptists from idaho to assist her in setting up an orphanage for haitian children.", "head_entity": "her", "tail_entity": "Idaho", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}
```

```json
{"index": 652, "sentence": "she owns been jailed in the eastern province of punjab since her initial arrest 1 1 / 2 years ago.", "head_entity": "She", "tail_entity": "Punjab", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 654, "sentence": "a haitian justice during saturday ordered the unleash from eight from the 10 ours detained here on child abduction charges , but he ruled that two members of the group , including its leader , the idaho businesswoman laura silsby , remain in jail so he can question them over reports that they had traveled to haiti before the jan 12 earthquake.", "head_entity": "Laura Silsby", "tail_entity": "Idaho", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}
```

```json
{"index": 658, "sentence": "one bar throughout the clans argued that puello took told him that he was licensed to practice law in florida , but the lawyer said he had checked and found no such record.", "head_entity": "he", "tail_entity": "Florida", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 669, "sentence": "augmented toward gays , it went to connecticut in his late 20s after eight years working in manhattan as an apprentice at a printer and as a store detective in times square -- a job he said he hated.", "head_entity": "his", "tail_entity": "Connecticut", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 678, "sentence": "lomax recalls where , through yr 13 , her mama moved him and his five brothers and sisters from los angeles , california , to tuskegee , alabama , to cover the civil rights movement for nation magazine.", "head_entity": "Lomax", "tail_entity": "Alabama", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 688, "sentence": "the appearances have meanwhile incited exam of silsby in idaho , exposing a complicated financial history with lawsuits pending against her and a house lost to foreclosure.", "head_entity": "Silsby", "tail_entity": "Idaho", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}
```

```json
{"index": 699, "sentence": "lomax remembers where , through yr 13 , her mummy moved him and his five brothers and sisters from los angeles , california , to tuskegee , alabama , to cover the civil rights movement for nation magazine.", "head_entity": "his", "tail_entity": "California", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}
```

```json
{"index": 701, "sentence": "lomax resembles when , at age 13 , his mother moved him and his five brothers and sisters from los angeles , california , to tuskegee , alabama , to cover the civil rights movement for nation magazine.", "head_entity": "his", "tail_entity": "California", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}
```

```json
{"index": 703, "sentence": "under hostages essentially never noted throughout liberate on wednesday was alan gross of potomac , md", "head_entity": "Alan Gross", "tail_entity": "Md", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 708, "sentence": "she continues in stable condition at the cleveland clinic in ohio .", "head_entity": "She", "tail_entity": "Ohio", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 722, "sentence": "toward malaysian during 9th nor 10th juni ' 07 the blog makes wwwqatalyscom seekers should be singaporean , others citizens or hold relevant residence status.", "head_entity": "The", "tail_entity": "wwwqatalyscom", "predicted_relation": "org:website", "predicted_relation_id": 9}
```

```json
{"index": 735, "sentence": "increasing up on oahu in the 1970s , she hung out at the scientology community church , or `` org , '' in downtown honolulu .", "head_entity": "she", "tail_entity": "Honolulu", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 740, "sentence": "the courthouse also delivered verdicts on three former chongqing police officers -- huang daiqiang , zhao liming and chen tao -- in the same hearing.", "head_entity": "Chen Tao", "tail_entity": "Chongqing", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 746, "sentence": "hartford sate ' s attorney david cohen said monday that there is no evidence that sandra herold of stamford was aware of risk that her chimpanzee posed to other people and disregarded it.", "head_entity": "Sandra Herold", "tail_entity": "Stamford", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 750, "sentence": "the stoppage against wen qiang , the former head of chongqing ' s judiciary and an ex - deputy police chief , brought to a climax a series of trials that lifted the lid on the seamy underworld in the mega - city of more than 30 million.", "head_entity": "Wen Qiang", "tail_entity": "Chongqing", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 757, "sentence": "the lower cour also delivered verdicts on huang daiqiang , zhao liming and chen tao , the three former chongqing police officers , in the same trial.", "head_entity": "Chen Tao", "tail_entity": "Chongqing", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 759, "sentence": "during one , he tours his suburban pittsburgh home , starting outside and moving inside the two - bedroom , brick house.", "head_entity": "his", "tail_entity": "Pittsburgh", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 761, "sentence": "tamaihia , whom did situated in walker ' s south sacramento home last month , died on monday.", "head_entity": "Tamaihia", "tail_entity": "Sacramento", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 769, "sentence": "silsby , 40 , among boise , idaho , was asked if she did n ' t consider it naive to cross the border without adoption papers at a time when haitians are so concerned about child trafficking.", "head_entity": "she", "tail_entity": "Boise", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 771, "sentence": "temperature did interned toward augusta and accused of protecting the gang operations masterminded by his sister - in - law , xie caiping , 46 , known as the `` godmother '' of the chongqing underworld.", "head_entity": "Xie Caiping", "tail_entity": "Chongqing", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 780, "sentence": "lawyers during the chongqing no. 5 intermediate people ' s court said wen qiang took bribes worth more than 16 million yuan - lrb - $ 2. 4 million - rrb - from gang members he gave legal protection while he was director of the chongqing municipal judicial bureau , the official xinhua news agency reported.", "head_entity": "Wen Qiang", "tail_entity": "Chongqing", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 783, "sentence": "the undated video apparently recorded by george sodini and posted on starcasmnet show him touring his suburban pittsburgh home.", "head_entity": "George Sodini", "tail_entity": "Pittsburgh", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 789, "sentence": "obvious , a 60 - sunni - immemorial native among potomac , maryland , was working for a firm contracted by usaid when he was arrested dec 3 , 2009 , and sent to cuba ' s high - security villa marista prison.", "head_entity": "he", "tail_entity": "Potomac", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 792, "sentence": "the 62 - sunni - immemorial vagni , whom did publicized july 12 , returned wednesday to her muscat of montevarchi , which welcomed him with an open - air party in a piazza.", "head_entity": "Vagni", "tail_entity": "Montevarchi", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 793, "sentence": "sodini graduated among pittsburgh in 1992 and named the school as beneficiary in a 2007 will.", "head_entity": "Sodini", "tail_entity": "Pittsburgh", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 804, "sentence": "nation ' s counsel dawood cohan argued thu that neither fees are planned against sandra herold , of stamford .", "head_entity": "Sandra Herold", "tail_entity": "Stamford", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 806, "sentence": "the dutch newsletter brabants dagblad argued the mec were probably from tilburg in the southern netherlands and that he had been on safari in south africa with his mother trudy , 41 , father patrick , 40 , and brother enzo , 11.", "head_entity": "his", "tail_entity": "Tilburg", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 812, "sentence": "arranged crime boss xie caiping , labeled the `` godmother of the underworld '' in southwest china ' s chongqing municipality , was sentenced 18 years in prison tuesday.", "head_entity": "Xie Caiping", "tail_entity": "Chongqing Municipality", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 815, "sentence": "for tuesday , organized crime boss xie caiping , labeled the `` godmother of the underworld '' in chongqing , was sentenced to 18 years in prison.", "head_entity": "Xie Caiping", "tail_entity": "Chongqing", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 826, "sentence": "increased in queens , he moved to connecticut in his late 20s after eight years working in manhattan as an apprentice at a printer and as a store detective in times square -- a job he said he hated.", "head_entity": "he", "tail_entity": "Queens", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 831, "sentence": "posed in queens , he moved to connecticut in his late 20s after eight years working in manhattan as an apprentice at a printer and as a store detective in times square -- a job he said he hated.", "head_entity": "his", "tail_entity": "Queens", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 855, "sentence": "the haya , might 14 , 2010 - lrb - afp - rrb - the laddie , identified by the ministry only as nine - year - old ruben from the southern city of tilburg but fully by the media as ruben van assouw , is recovering in a tripoli hospital after surgery on his smashed legs.", "head_entity": "Ruben van Assouw", "tail_entity": "Tilburg", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 856, "sentence": "onto the 14 years after her daughter was born , hagemo had a second daughter , divorced and moved near her mother in minneapolis .", "head_entity": "her", "tail_entity": "Minneapolis", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 859, "sentence": "giuca did impeached from the 2003 assassinate of mark fisher , a 19 - year - old college student from andover , new jersey , who had gone to an after - hours party giuca hosted in brooklyn while his parents were out of town.", "head_entity": "Mark Fisher", "tail_entity": "Andover", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}
```

```json
{"index": 862, "sentence": "sodini graduated toward 1992 for the campus from cincinnati with a degree in computer science and had worked as a systems analyst at a pittsburgh law firm since 1999.", "head_entity": "Sodini", "tail_entity": "Pittsburgh", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 864, "sentence": "lomax exchanges a story about almena lomax , his mother and a newspaper owner and journalist in los angeles , taking her family on the bus to tuskegee , ala. , in 1961.", "head_entity": "his", "tail_entity": "Los Angeles", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 866, "sentence": "the dutch newsletter brabants dagblad explained ruben was probably from tilburg in the southern netherlands and that he had been on safari in south africa with his mother trudy , 41 , father patrick , 40 , and brother enzo.", "head_entity": "Ruben", "tail_entity": "Tilburg", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 870, "sentence": "lawyers argued inquiries showed wen had abused his power by offering benefits to others and accepted huge bribes personally or with his wife zhou xiaoya when he served in a series of communist party of china and administrative posts at the public security bureau of chongqing .", "head_entity": "he", "tail_entity": "Chongqing", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 877, "sentence": "the ltalian court came under harsh criticism from many who felt knox , a native of seattle , washington , was convicted only on circumstantial evidence and that the italian press , which jurors were free to read , had assassinated her character.", "head_entity": "Knox", "tail_entity": "Seattle", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}
```

```json
{"index": 878, "sentence": "ecoffey warned juries that he and another federal agent met with graham in april 1994 in yellowknife , the city in northwest canada where graham lived at the time.", "head_entity": "Graham", "tail_entity": "Yellowknife", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 879, "sentence": "the prosecutor denied to expose the nickname from the mec , whom hails for the southern township among tilburg , to protect the family ' s privacy , but a regional daily newspaper said he was probably ruben van assouw .", "head_entity": "Ruben van Assouw", "tail_entity": "Tilburg", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 884, "sentence": "since booting , the laddie identified by the foreign ministry only as ruben from tilburg in the southern netherlands and more fully as ruben van assouw by dutch media , was taken by ambulance with two doctors and a nurse to hospital.", "head_entity": "Ruben van Assouw", "tail_entity": "Tilburg", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 885, "sentence": "it warned him it became ruben , nine years old , from the city of tilburg , '' prommersberger said.", "head_entity": "Ruben", "tail_entity": "Tilburg", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 891, "sentence": "yet , the booklet focuses during knox ' s personage , his infantile in seattle , her hopes for post - prison life.", "head_entity": "Knox", "tail_entity": "Seattle", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}
```

```json
{"index": 892, "sentence": "in prisoners notably not mentioned for release on wednesday was alan gross of potomac , md", "head_entity": "Alan Gross", "tail_entity": "Potomac", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 894, "sentence": "never extraordinarily many pastors are that committed to their ministry , '' elder steve taplin said of daniels , who traveled 60 miles from oklahoma city every sunday to the worthy temple christ holy sanctified church in anadarko.", "head_entity": "Daniels", "tail_entity": "Oklahoma City", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 909, "sentence": "the juni. 16 aggression appeared where the 200 - libra - lrb - 90 - kilogram - rrb - animal ' s owner , sandra herold , asked nash , her friend and employee , to help lure the animal back into her house in stamford , connecticut.", "head_entity": "Sandra Herold", "tail_entity": "Stamford", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 912, "sentence": "the 21 - annum - old victim , meredith kercher , was in perugia as an exchange student from leeds university and shared a cottage with seattle native knox .", "head_entity": "Knox", "tail_entity": "Seattle", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}
```

```json
{"index": 915, "sentence": "sheriff roberta ecoffey nor board from aboriginal relations unusual constable mich pourier said saturday that they met among graham among april 1994 in the western canada town of whitehorse , where graham lived at the time.", "head_entity": "Graham", "tail_entity": "Whitehorse", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 923, "sentence": "during thu nor wednesday , sodini took off from his job as a systems analyst at a pittsburgh law firm -- where he ' d worked since 1999 -- `` to practice my routine and make sure it is well polished , '' he wrote in his diary.", "head_entity": "he", "tail_entity": "Pittsburgh", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 924, "sentence": "`` j learn for cadets nor i challenge them , '' says heloise , 58 , whom became over the dynasty councils undertaking where his mamma , also named heloise , died in 1977.", "head_entity": "Heloise", "tail_entity": "Heloise", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 925, "sentence": "the dutch newsletter brabants dagblad argued the mec did supposedly for tilburg toward the southern antilles and that he had been on safari in south africa with his mother trudy , 41 , father patrick , 40 , and brother enzo , 11.", "head_entity": "he", "tail_entity": "Patrick", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 928, "sentence": "victims include his wife , sandra ; four sons , jeff , james , douglas and harris ; a daughter , leslie ; his mother , sally ; and two brothers , guy and paul.", "head_entity": "his", "tail_entity": "Sally", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 929, "sentence": "knox ' s mamma edda mellas was shaking with grief as her stepmother cassandra knox moved to comfort her.", "head_entity": "Knox", "tail_entity": "Edda Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 932, "sentence": "coulter nor his papa , across among silsby ' s pater john sander , traveled to haiti on may 2 to give silsby moral support.", "head_entity": "Silsby", "tail_entity": "John Sander", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 941, "sentence": "knox impeached , ruled to 26 decades toward ltaly the american ' s pater , curt knox , asked if he would fight on for his daughter , replied , with tears in his eyes : `` hell , yes. ''", "head_entity": "Knox", "tail_entity": "Curt Knox", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 945, "sentence": "`` j learn for cadets nor i challenge them , '' says heloise , 58 , whom became over the dynasty councils undertaking where his mamma , also named heloise , died in 1977.", "head_entity": "Heloise", "tail_entity": "Heloise", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 947, "sentence": "she ' s been through this throughout stronger upon 30 decades nor learned for the highest , his mamma , eloise bowles , who wrote the first hints from heloise column for 18 years.", "head_entity": "She", "tail_entity": "Eloise Bowles", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 952, "sentence": "terni , ltaly 2010 - 12 - 18 10 : 17 : 47 utc knox ' s mama , edda mellas , was in court saturday.", "head_entity": "Knox", "tail_entity": "Edda Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 959, "sentence": "the american ' s pater , curt knox , asked if he would fight on for his daughter , replied , with tears in his eyes : `` hell , yes. ''", "head_entity": "The American", "tail_entity": "Curt Knox", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 961, "sentence": "horrendous ny pleased throughout spanish commuters : gendarmerie pursuit mamane nor his mamma , claudine , traveled to new york to attend a religious seminar.", "head_entity": "Mamane", "tail_entity": "Claudine", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 968, "sentence": "toward a explanation leen to the investigative sworn , jupp ' s relatives roy and anne said their son had been involved in vital work -- but had likely been failed by an inadequate safety regime.", "head_entity": "Jupp", "tail_entity": "Anne", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 987, "sentence": "`` folks are launching to acquire quite what this incident makes admittedly almost -- which makes an asia from false conviction , '' knox ' s stepfather , chris mellas , told reporters in perugia.", "head_entity": "Knox", "tail_entity": "Chris Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 999, "sentence": "toward complement to john sauvage nor lien , victims include her stepchildren , susan , danielle , elizabeth , kate , michael , stephen and maggie mailer ; and her mother , gaynell davis .", "head_entity": "her", "tail_entity": "Gaynell Davis", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 1002, "sentence": "mamane nor his mamma , claudie , traveled to new york to attend a religious seminar.", "head_entity": "Mamane", "tail_entity": "Claudie", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 1006, "sentence": "`` we any are optimistic nor we trust those tribunals nor the sworn to know that they are leaving to never put two innocuous minors toward inmate throughout a felon that they had n ' t perpetrate , '' knox ' s mama , edda mellas , told reporters after the hearing.", "head_entity": "Knox", "tail_entity": "Edda Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 1007, "sentence": "knox ' s pater , curt knox , told the cbs `` early show '' from seattle that his ex - wife ' s testimony would strengthen his daughter ' s case.", "head_entity": "Knox", "tail_entity": "Curt Knox", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 1008, "sentence": "edda mellas , knox ' s mummy , said the case has been taking a toll on her daughter. ''", "head_entity": "Knox", "tail_entity": "Edda Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 1031, "sentence": "for wednesday , ouattara ' s newly appointed un envoy , youssoufou bamba , said that the country is on the `` brink of genocide , '' citing reports that pro - gbagbo gangs have been marking the homes of members of ethnic groups allied with ouattara.", "head_entity": "Youssoufou Bamba", "tail_entity": "UN", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1035, "sentence": "it named one because shah abdul aziz , a member of a pro - taliban religious party who was elected to parliament ' s lower house in 2002 as part of an anti - american alliance made up of several religious parties.", "head_entity": "Shah Abdul Aziz", "tail_entity": "parliament", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1045, "sentence": "problematic sunni throughout fille from boy interned because bugging across cuba his work was part of a program of the us agency for international development , a government agency that provides economic and humanitarian assistance worldwide but has also been criticized by cuba for seeking to promote democratic change on the island.", "head_entity": "His", "tail_entity": "Development", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1047, "sentence": "`` there ' s a recent confess that there makes opulence out there in the african - american community and that the colleges have a strong claim on it , '' said michael lomax , president of the united negro `` third thursday '' < tthurs @ gmailcom >", "head_entity": "Michael Lomax", "tail_entity": "United Negro", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1048, "sentence": "vagni got one of three workers from the international committee of the red cross who were kidnapped on jolo , an impoverished muslim region 590 miles - lrb - 950 kilometers - rrb - south of manila , on jan 15.", "head_entity": "Vagni", "tail_entity": "International Committee of the Red Cross", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1056, "sentence": "shackle , june 21 - lrb - xinhua - rrb - three members of the international committee of the red cross - lrb - icrc - rrb - -- swiss andreas notter , italian eugenio vagni and filipina mary jean lacaba -- were kidnapped in patikul town of sulu province on jan 15 after the team conducted a humanitarian mission there.", "head_entity": "Eugenio Vagni", "tail_entity": "the International Committee of the Red Cross", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1066, "sentence": "`` this makes the trajectory the detail makes leaving , '' argued alessi , whom makes jefe of neurology at the william w. backus hospital in norwich , conn. `` we need to take this study very seriously. ''", "head_entity": "Alessi", "tail_entity": "William W. Backus Hospital", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1070, "sentence": "she meanwhile collaborated at florida state university when that team reached the 2005 us college semifinals.", "head_entity": "She", "tail_entity": "Florida State University", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1072, "sentence": "the us vlado owns ascertained the boy , 60 - anno - old alan gross , as a government contractor with development alternatives inc - lrb - dai - rrb - of bethesda , maryland , just outside washington , who was seeking to help jewish groups communicate with people outside the country by distributing mobile phones and computers.", "head_entity": "Alan Gross", "tail_entity": "Development Alternatives Inc", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1075, "sentence": "adela dworin , presidency of havana ' s temple beth shalom and cuba ' s largest jewish organization , the jewish community house , told the associated press on wednesday it ' s possible gross came to the center as one of `` hundreds '' of foreign visitors it receives each year.", "head_entity": "Adela Dworin", "tail_entity": "Jewish Community House", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1082, "sentence": "the constabulary and marines were hunting for the abu sayyaf band holding 62 - year - old red cross worker eugenio vagni when they were attacked by dozens of gunmen on the island of jolo , military and police spokesmen said.", "head_entity": "Eugenio Vagni", "tail_entity": "Red Cross", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1089, "sentence": "onto 2006 , he became a senior research scientist at nasa ' s jet propulsion laboratory and was appointed chairman of caltech ' s physics , mathematics and astronomy division in 2008.", "head_entity": "he", "tail_entity": "Caltech", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1093, "sentence": "auditor ' s biologic : wayne a. holst learns religion and culture at the university of calgary and helps facilitate adult spiritual development at st. david ' s united church , calgary.", "head_entity": "Wayne A. Holst", "tail_entity": "University of Calgary", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1097, "sentence": "a sentence on the case of wen qiang , former justice department director in the southwestern city of chongqing , would be announced at an undisclosed , later date , the no", "head_entity": "Wen Qiang", "tail_entity": "justice department", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1103, "sentence": "toward recent ny during saturday , ivory rive ' s recent un ambassador , youssoufou bamba , said he was worried about his country ' s future and was consulting with members of the security council ahead of a meeting next week on ways to help ouattara assume power.", "head_entity": "Youssoufou Bamba", "tail_entity": "UN", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1108, "sentence": "nor that the institutes have a solid request during he , '' argued michael lomax , president of the united negro", "head_entity": "Michael Lomax", "tail_entity": "United Negro", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1112, "sentence": "tim will take leader from a vastly talented computer from nba servants based in china including mark fisher , who will be promoted to senior vice president , '' said heidi ueberroth , nba president of global marketing partnerships and international business operations.", "head_entity": "Mark Fisher", "tail_entity": "NBA", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1118, "sentence": "parad ' s ribbon snatched filipina mary jean lacaba , swiss national andreas notter and italian eugenio vagni in january last year while they were on a jolo humanitarian mission for the international committee of the red cross .", "head_entity": "Eugenio Vagni", "tail_entity": "International Committee of the Red Cross", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1119, "sentence": "miettinen did an helper at franklin pierce college in 1999 , helping her alma mater to its fourth national championship in five years.", "head_entity": "Miettinen", "tail_entity": "Franklin Pierce College", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1121, "sentence": "detectives argued that among thu ' s arrests they are closing in on matteo messina denaro , a fugitive who is among a handful of mobsters vying to take over the sicilian mafia .", "head_entity": "Matteo Messina Denaro", "tail_entity": "Sicilian Mafia", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1139, "sentence": "adela dworin , presidency of havana ' s temple beth shalom and cuba ' s largest jewish organization , the jewish community house , told the associated press it ' s possible gross came to the center as one of hundreds of foreign visitors it receives each year.", "head_entity": "Adela Dworin", "tail_entity": "Jewish Community House", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1144, "sentence": "ble goude did birthed onto 1972 in gbagbo ' s centre west home region , guiberoua , and rose to become secretary general of the powerful and aggressive students ' federation of ivory coast - lrb - fesci - rrb - .", "head_entity": "Ble Goude", "tail_entity": "Students ' Federation of Ivory Coast", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1147, "sentence": "j did an e - courier currency among benjamin chertoff among popular mechanics in the original loose change thread that showed that he was not a close relative of michael chertoff.", "head_entity": "Benjamin Chertoff", "tail_entity": "Popular Mechanics", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1148, "sentence": "clad toward a diary drizzle overcoat and dungarees that accentuated his weight loss , he recounted his time in captivity after he and two other red cross aid workers were kidnapped on january 15.", "head_entity": "he", "tail_entity": "Red Cross", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1162, "sentence": "miettinen hired per wps champ sky blue fc", "head_entity": "Miettinen", "tail_entity": "Sky Blue FC", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1205, "sentence": "it served as vice director of the chongqing municipal public security bureau from 1992 to 2008 , and later as director of the chongqing municipal judicial bureau until he was arrested in september 2009.", "head_entity": "he", "tail_entity": "Chongqing Municipal Public Security Bureau", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1207, "sentence": "ivory rive ' s recent un envoy , youssoufou bamba , said he is worried about his country ' s future and is consulting with members of the security council ahead of a meeting next week on ways to help ouattara assume power.", "head_entity": "Youssoufou Bamba", "tail_entity": "UN", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1208, "sentence": "eugenio vagni , the ltalian workman of the icrc , and colleagues andreas notter of switzerland and mary jean lacaba of the philippines were released by their abu sayyaf captors separately.", "head_entity": "Eugenio Vagni", "tail_entity": "ICRC", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1211, "sentence": "two from jupp ' s managers at the government ' s military science service -- the defense , scientific and technology laboratory -- were charged with manslaughter following an inquiry into the blast.", "head_entity": "Jupp", "tail_entity": "Defense , Scientific and Technology Laboratory", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1238, "sentence": "but piedra perpetually seemed to have money for his favorite cause : the church of scientology .", "head_entity": "Piedra", "tail_entity": "Church of Scientology", "predicted_relation": "per:religion", "predicted_relation_id": 24}
```

```json
{"index": 1242, "sentence": "peshawar governments have arrested a former member of parliament in connection with the murder of a polish engineer decapitated by taliban militants in february , a security official said sunday.", "head_entity": "member", "tail_entity": "parliament", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1251, "sentence": "onto august , baldino , who lived in west chester , pa. , had taken a leave of absence from his posts as cephalon ' s chairman and chief executive.", "head_entity": "Baldino", "tail_entity": "Cephalon", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1263, "sentence": "two from jupp ' s leaders at the defense , scientific and technology laboratory were charged with manslaughter following an inquiry into the blast.", "head_entity": "Jupp", "tail_entity": "Defense", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1269, "sentence": "philippine mainland registrar xabi puno argued during saturday there are through weakest six hijackers whom remain toward the mains from the takers toward southern islanders from solo nor basilan , embracing ltalian nacional eugenio vagni , the 62 - sunni - immemorial engineering of the international committee of the red cross .", "head_entity": "Eugenio Vagni", "tail_entity": "International Committee of the Red Cross", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1271, "sentence": "guaranteeing freedoms in alberta an article prepared for `` sightings '' produced at the martin marty centre , university of chicago divinity school by wayne a. holst - lrb - 750 words - rrb -", "head_entity": "Wayne A. Holst", "tail_entity": "University of Chicago Divinity School", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1342, "sentence": "ahmet mukhlis jozef , massa from jakarta ' s nacional journalist entity antara , was elected president of the organization of asia pacific news agencies - lrb - oana - rrb - at the organization ' s 13th general assembly taking place here wednesday.", "head_entity": "Organization of Asia Pacific News Agencies", "tail_entity": "Antara", "predicted_relation": "org:members", "predicted_relation_id": 22}
```

```json
{"index": 1692, "sentence": "advocates nor lawyers contended toward justice that , relying to negotiations furtively included per a trip per mellas to his jailed daughter , mellas indicated that knox had already spoken during her first call home about a foot having been seen in kercher ' s bedroom -- before the body was discovered.", "head_entity": "her", "tail_entity": "Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

所有样本分析完毕。