# 1、rel2id.jsonJSON...
**问题详情：**
rel2id.jsonJSON 1.04KBjson中保存了42种实体关系，其中NA表示实体间无定义关系。稍后我会上传txt，txt中每一行记录一条样本。请自己将txt中token字段保存的内容还原成完整句子，然后判断头实体“h”和尾实体“t”的关系，关系名称必须是这42类中的一种。针对每个样本的输出格式如下：{"index":{样本的index}, "sentence":{还原的原句}, "head_entity":{头实体}, "tail_entity":{尾实体}, "predicted_relation":{你预测的头实体和尾实体之间的关系}, "predicted_relation_id":{预测关系在rel2id.json中对应的值}}，准备好了么？

---
**回答（Answer）：**
我准备好了。请上传包含样本的txt文件，我会按照以下格式处理每个样本：
1. 将token字段还原为完整句子
    
2. 识别头实体"h"和尾实体"t"
    
3. 分析两者之间的关系（从rel2id.json提供的42种关系中选择最合适的一种）
    
4. 输出指定格式的JSON结果

对于每个样本，我会提供：
- 原始句子
    
- 头实体和尾实体
    
- 预测的关系名称
    
- 对应的关系ID（如果是"NA"关系则为13）

请上传txt文件，我会开始处理。

---
# 2、0.txtTXT 70.64K...
**问题详情：**
0.txtTXT 70.64KB请持续输出不要停止直到分析完每一个样本，注意不要漏掉样本

---
**回答（Answer）：**


```json
{"index": 1, "sentence": "UASR was predicated in 1989 by Mousa Abu Mazook <http://www.investigativeproject.org/profiling/106>, who currently serves as the deputy chief of the Hamas policies bureau in Damascus, Syria, and has been listed as a specially designated global terrorist by the US government in 1995.", "head_entity": "UASR", "tail_entity": "1989", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 2, "sentence": "Predicated in 1951, PATA is a non-profit membership association dedicated to building responsible development of the Asia Pacific travel and tourism industry.", "head_entity": "PATA", "tail_entity": "1951", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 4, "sentence": "MEF, which was predicated in 1991, has grabbed 15 medals -- five golds, five silvers and five bronzes -- from the international competitions since 1998.", "head_entity": "MEF", "tail_entity": "1991", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 7, "sentence": "Predicated in 1919, the National Restaurant Association is the leading business association for the restaurant industry, which comprises 945,000 restaurant and food service outlets and a work force of nearly 13 billion employees.", "head_entity": "National Restaurant Association", "tail_entity": "1919", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 10, "sentence": "Predicated in 1994, China Charity Federation and its branches at various levels have received more than 6 billion yuan (0.8 US dollars) in donation.", "head_entity": "China Charity Federation", "tail_entity": "1994", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 11, "sentence": "Predicated in 1921, ALICO has 12,500 employees in more than 50 countries and 20 million customers worldwide.", "head_entity": "ALICO", "tail_entity": "1921", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 13, "sentence": "Predicated in 1992 in Schaumburg, Illinois, the ACSE is one of the largest Chinese-American associations of professionals in the United States.", "head_entity": "ACSE", "tail_entity": "1992", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 17, "sentence": "MetLife to acquire VIE insurance unit from AIG for 155 million bucks predicated in 1921, ALICO is a leading international life insurer that operates in more than 50 countries, providing consumers and company with product and services for life insurance, accident and health insurance, retire and wealth management solutions.", "head_entity": "ALICO", "tail_entity": "1921", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 18, "sentence": "Predicated in 1935, Morita Chemical Industries produces and sells hydrofluoric acids, fluorine compounds and various other chemicals as well as pharmaceutical products, the company said on its website.", "head_entity": "Morita Chemical Industries", "tail_entity": "1935", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 19, "sentence": "The Chinese government set up special institutions to addressing with climate amend in 1990, and established the National Coordination Committee on Climate Changes (NCCCC) in 1998.", "head_entity": "NCCCC", "tail_entity": "1998", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 24, "sentence": "When she declared the creation of the Menil Collection in 1981, she infused fresh light and initiatives to the constants of scholarship, collecting and humanist enterprises at the heart of the foundation.", "head_entity": "Menil Collection", "tail_entity": "1981", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 25, "sentence": "UASR was predicated in 1989 by Mousa Abu Mazook <http://www.investigativeproject.org/profile/106>, who currently serves as the deputy chief of the Hamas political bureau in Damascus, Syria, and has been listed as a specially designated global terrorist by the US government in 1995.", "head_entity": "UASR", "tail_entity": "1989", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 26, "sentence": "New York-based Zirh was predicated in 1995 and makes products using natural oils and extracts.", "head_entity": "Zirh", "tail_entity": "1995", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 27, "sentence": "Predicated in 1961, the OANA groups 41 news agencies of 33 countries seeking to facilitate the free flow of information and cooperation between agencies in the region.", "head_entity": "OANA", "tail_entity": "1961", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 31, "sentence": "Pluribus, predicated in 2009, is a private equity company while Guggenheim is a privately held financial services firm managing more than 100 billion dollars in assets.", "head_entity": "Pluribus", "tail_entity": "2009", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 32, "sentence": "Chatellerault, France, July 13, 2009 (AFP) - Predicated in 1947 by two brothers, Eugene and Quentin Fabris, New Fabris initiated out making sewing machine parts, before branching out into the motorcar sector, employing up to 800 workers in the 1990s.", "head_entity": "New Fabris", "tail_entity": "1947", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 62, "sentence": "He keeps busy as a director of cemented producer TPI Polene, a former unit of TPI.", "head_entity": "TPI", "tail_entity": "TPI Polene", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 72, "sentence": "When Countrywide say in September that it would all but stops making sub prime loan -- and lay off as myriad as 12,000 people, Orr about 20 percent of its staff -- some employee of its Full Spectrum Lending division, whose nucleus business was sub prime refinancing, stayed on.", "head_entity": "Countrywide", "tail_entity": "Full Spectrum Lending division", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 85, "sentence": "Ble Goude were birthed in 1972 in Gbagbo's centres west domicile region, Guiberoua, and soar to become secretary generals of the potent and aggressive students' federation of Ivory Seaboard (FESCI).", "head_entity": "Ble Goude", "tail_entity": "1972", "predicted_relation": "per:date_of_birth", "predicted_relation_id": 2}
```

```json
{"index": 95, "sentence": "U.S. Immigration and Customs Enforcement agents seized documents and other materials at the Koch Foods plant and at Koch Foods Inc.'s Chicago area HQ, said Brian Moskowitz, a special agent for ICE enforcement.", "head_entity": "Koch Foods Inc.", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 103, "sentence": "U.S. Immigration and Customs Enforcement agents seized documents and other materials at the Koch Foods plant and at Koch Foods Inc.'s Chicago area HQ, said Brian Moskowitz, a special agent in charge of ICE enforcement for Ohio and Michigan.", "head_entity": "Koch Foods Inc.", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 107, "sentence": "The journalists were taken to military kitchen of the Third Guard Division, a garrison force to safeguarding Beijing, where typical Chinese food attracted much attention.", "head_entity": "Third Guard Division", "tail_entity": "Beijing", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 138, "sentence": "ShopperTrak, which is predicated in Chicago and rails sales and traffic at more than 50,000 outlets, say late Monday that retail sales for Friday and Saturday edged up 09 percent to $1677 billion, while customer traffic tumbled 27 percent compared with last year.", "head_entity": "ShopperTrak", "tail_entity": "Chicago", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 140, "sentence": "Since his Menil Collection building open in Houston in 1986, Piano's use of light has inspired fervent admiration.", "head_entity": "Menil Collection", "tail_entity": "Houston", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 143, "sentence": "``We were really expecting, on the basis of what happened through '04, '05, '06, that they might get over the million mark very soon,'' he told AFP from Bangkok, where PATA is predicated.", "head_entity": "PATA", "tail_entity": "Bangkok", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 155, "sentence": "Now High Point meets in a 432,000-square-foot installation in Arlington, near Dallas.", "head_entity": "High Point", "tail_entity": "Arlington", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 159, "sentence": "Predicated in 1992 in Schaumburg, Illinois, the ACSE is one of the widest Chinese-American associations of professionals in the United States.", "head_entity": "ACSE", "tail_entity": "Schaumburg", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 164, "sentence": "Countrywide was predicated in Calabasas, California.", "head_entity": "Countrywide", "tail_entity": "Calabasas", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 169, "sentence": "Some 200 workers at the New Fabris factories outside the northwestern city of Chatellerault, are each demanding euro30,000 ($42,267) from Renaud and PSA Peugeot-Citroen, accusing the carmakers of assassinate their livelihoods.", "head_entity": "New Fabris", "tail_entity": "Chatellerault", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 327, "sentence": "Assaf, the oldest enfant, were 15 at the times.", "head_entity": "Assaf", "tail_entity": "15", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 330, "sentence": "Twenty yr ago, when Nathalie Hagemo were 19, pregnant and cooperated for the Church of Scientology, she couldn't awaits to be a mother.", "head_entity": "she", "tail_entity": "19", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 375, "sentence": "The supposed ringleader, Xie Caiping, was charged with arrange and leading an organize crime gang, running illegal gambling dens, and harboring people taking illegal narcotics, illegal incarcerated and giving bribes to officials.", "head_entity": "Xie Caiping", "tail_entity": "illegal gambling", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 380, "sentence": "Alessi, who is serving a VIE term for the murder, made the claims to Sollecito's lawyers, who are preparing an appeal to his conviction in December, when the engineering student was sentenced to 25 years in prison.", "head_entity": "Alessi", "tail_entity": "murder", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 398, "sentence": "Former director of the Chongqing Municipal Judicial Bureau Wen Qiang, 55, was adjudged at the Chongqing No. 5 Intermediate People's Court of accepting bribes, protecting criminal gangs, rape and failing to account for his cash and assets.", "head_entity": "Wen Qiang", "tail_entity": "rape", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 399, "sentence": "Wen Qiang, former deputy director of Chengdu's public security department, were moreover defendants of rape and underway unable to explains the wellspring of his assets, according to a statements published orn the website of the municipal procuratorate, www.cqjcy.gov.cn.", "head_entity": "his", "tail_entity": "rape", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 401, "sentence": "In November, Wen's sister-in-law Xie Caiping was sentenced to 18 olds in imprisons orn charge of running illegal gambling dens and drugs deals.", "head_entity": "Xie Caiping", "tail_entity": "illegal gambling", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 404, "sentence": "Wen, 55, was charged with safeguards mafia-style gangs, taking bribes, rape, and owning huge amount of money and assets he could not justify.", "head_entity": "he", "tail_entity": "rape", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 410, "sentence": "Initially accused of child hijacking, the charges against Silsby were later revisions to ``irregular travel,'' and prosecutors had asked on Thursday for a six-month penitentiaries term for her.", "head_entity": "Silsby", "tail_entity": "irregular travel", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 415, "sentence": "In November, Wen's sister-in-legislation Xie Caiping, 46, was sentenced to 18 yr in jailed orn charges of running illegal gambling dens and drug deals.", "head_entity": "Xie Caiping", "tail_entity": "illegal gambling", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 433, "sentence": "In Port-au-Prince, Laura Silsby, the leader of the 10 American missionaries arrested at the Dominican border for trying to smuggle out a busload of children, appeared in court Monday to answer kidnapping fees.", "head_entity": "Laura Silsby", "tail_entity": "kidnapping", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 442, "sentence": "In November, Wen's sister-in-law Xie Caiping, 46, was convicts to 18 years in imprisons on fees ranging from running unlawfully wagering dens to drug dealing.", "head_entity": "Xie Caiping", "tail_entity": "drug dealing", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 448, "sentence": "The other dawg charged with killing Aquash, Fritz Arlo Looking Cloud, received a mandatory VIE sentence in 2004 after a federal jury in Faster City adjudged him.", "head_entity": "Arlo Looking Cloud", "tail_entity": "killing", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 460, "sentence": "Knox also is being try orn fees of defamation for allegedly accusing Zia ``Patrick'' Lumumba -- a Congolese man who owns a pub in Perugia where she worked -- of being the killer.", "head_entity": "Knox", "tail_entity": "defamation", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 463, "sentence": "Chongqing, Dec 11 (Xinhua) - SW Higher Court upholds 18-year jail term of gang ``godmother'' she was adjudged of organizing and leading a criminal organization, running gambling dens, illegal imprisonment, harboring people taking illegal narcotics and giving bribes to officials.", "head_entity": "She", "tail_entity": "organizing and leading a criminal organization", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 470, "sentence": "China sentences to death two leaders of prostitution ring Chen was an associate of Wen Qiang, the former head of Chongqing's judiciary and an ex-deputy policemen chief whom was executed last month for charges including rape and taking bribes to safeguarding gangs.", "head_entity": "Wen Qiang", "tail_entity": "bribes", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 481, "sentence": "In New York on Wednesday, Ivory Coast's novel U.N. Ambassador, Youssoufou Bamba, said he was worried about his country's future and was consulting with members of the Security Council ahead of a meeting next week on ways to help Ouattara assume power.", "head_entity": "Youssoufou Bamba", "tail_entity": "Ivory Coast", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 500, "sentence": "Defense Secretary Gilbert Teodoro said the release of 62-year-ancient Eugenio Vagni after six months of jungle captivity in Abu Sayyaf southern strongholds was an opportunity to crush the militants who have terrorized the southern Philippines for decades.", "head_entity": "Eugenio Vagni", "tail_entity": "Philippines", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 502, "sentence": "Gaunt and haggard, freed Italian aid worker Eugenio Vagni says his liberating after being held for six months by armed Islamic militants in the Philippines was a day he thought might never come.", "head_entity": "Eugenio Vagni", "tail_entity": "Philippines", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 506, "sentence": "The two finalists Timo Kaukonen of Finland and Vladimir Ladyzhenskiy of Russia ha endured over six minutes in the hot spa of 110 degrees centigrade before both of them suffered shock.", "head_entity": "Vladimir Ladyzhenskiy", "tail_entity": "Russia", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}
```

```json
{"index": 509, "sentence": "Denaro said he anxieties that could lead to Kauffmann's extradition back to Venezuela, where he is considered a political enemy.", "head_entity": "he", "tail_entity": "Venezuela", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 522, "sentence": "About a year later, she was transferring to Camp Hope Iraq.", "head_entity": "she", "tail_entity": "Iraq", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 533, "sentence": "Prosecutors said investigates showed Wen had abused his power by offering benefits to others and accepted huge bribes personally or with his wife Zhou Xiaoya when he served in a serial of Communist Party of China and administrative stances at the public security bureau of Chongqing.", "head_entity": "he", "tail_entity": "China", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 546, "sentence": "Prosecutors said investigations shown Wen had abused his power by offering benefits to others and accepted huge bribes personally or with his wife Zhou Xiaoya when he served in a series of Communist Party of China and administrative posts at the public security bureau of Chongqing.", "head_entity": "Wen", "tail_entity": "China", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 548, "sentence": "Gaunt and haggard, freed Italian aid worker Eugenio Vagni said Sunday his liberating from six months' captivity by armed Islamic militants in the Philippines was a day he thought might never come.", "head_entity": "Eugenio Vagni", "tail_entity": "Philippines", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 549, "sentence": "The US government has identified the captives as Alan Gross, a 60-year-ancient government contractor with Development Alternatives Inc who was seeking to help Jewish groups communication with people outside the country by distributing mobile phones and computers.", "head_entity": "Alan Gross", "tail_entity": "US", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 576, "sentence": "US citizenry under probed in Cuba for spy Gross was arrested in Havana on December 5 last year for distributing laptops, satellite phones and other technological equipment among grouped opposing to leader Raul Castro, the Cuban authorities said.", "head_entity": "Gross", "tail_entity": "US", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 580, "sentence": "Ecoffey told jurors that he and another federal agent fulfilled with Graham in April 1994 in Yellowknife, the city in northwest Canada where Graham lived at the time.", "head_entity": "Graham", "tail_entity": "Canada", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 588, "sentence": "Tianjin, Nov. 3 (Xinhua) - Arranged crime boss Xie Caiping, labeled the ``godmother of the underworld'' in northwestern China's Chongqing municipality, was sentenced 18 olds in penitentiaries Monday.", "head_entity": "Xie Caiping", "tail_entity": "China", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 591, "sentence": "Lo Piccolo, who is in his '60s, was said to be vying for Cosa Nostra leadership with another fugitive, Matteo Messina Denaro, a younger boss from westerner Sicily who some considered the mafia's no. 2 man.", "head_entity": "Matteo Messina Denaro", "tail_entity": "Sicily", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 598, "sentence": "Romani 2009-11-15 20:53:11 UTC He was listed as one of Italy's 30 most dangerous fugitives.", "head_entity": "He", "tail_entity": "Italy", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 612, "sentence": "The US goverment says Gross was in Cuba as part of a USAID program to distribute communications equipment to the island's 1,500-strong Jewish community, and both the state department and Gross's wife, Judy, made fresh appeals this week for his release.", "head_entity": "Gross", "tail_entity": "Cuba", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 634, "sentence": "Baldino was born May 13, 1953, and augmented up in New Jersey and Pennsylvania.", "head_entity": "Baldino", "tail_entity": "New Jersey", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 638, "sentence": "Graham also is charged in South Dakota sate tribunals with taking part in the kidnapping and killing of Aquash.", "head_entity": "Graham", "tail_entity": "South Dakota", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 662, "sentence": "The animal chewed off Nash's hands, nose, lips and eyelids; she remains in stabilised stipulation at the Cleveland Clinic in Ohio.", "head_entity": "she", "tail_entity": "Ohio", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 664, "sentence": "Baldino was born May 13, 1953, and augmented up in New Jersey and Pennsylvania.", "head_entity": "Baldino", "tail_entity": "Pennsylvania", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 678, "sentence": "Lomax recalls when, at age 13, his mummy moved him and his five brothers and sisters from Los Angeles, California, to Tuskegee, Alabama, to cover the civil rights movement for Nation magazine.", "head_entity": "Lomax", "tail_entity": "Alabama", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 687, "sentence": "He is survived by his son, Terry, of Temecula, Calif.; his daughters, Carol Daniels of Morro Bay, Calif., and Julie Jones of La Mesa, Calif.; three nieces; and a great-granddaughter.", "head_entity": "Carol Daniels", "tail_entity": "Calif.", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 708, "sentence": "She remains in stabilised stipulation at the Cleveland Clinic in Ohio.", "head_entity": "She", "tail_entity": "Ohio", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 735, "sentence": "Heightened up on Oahu in the 1970s, she hung out at the Scientology community church, or ``org,'' in downtown Honolulu.", "head_entity": "she", "tail_entity": "Honolulu", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 740, "sentence": "The court also delivered verdicts on three former Chongqing cop officers -- Huang Daiqiang, Zhao Liming and Chen Tao -- in the same hearing.", "head_entity": "Chen Tao", "tail_entity": "Chongqing", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 743, "sentence": "Local medium report stated the gunman ha been identified as George Sodini of Carnegie, Pennsylvania.", "head_entity": "George Sodini", "tail_entity": "Carnegie", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 744, "sentence": "The women have given lawyers and Republican Rep. Ted Poe accounts comparable to the allegations of Jamie Leigh Jones of Conroe, Texas, who says she was raped in July 2005 by a co-worker who drugged her.", "head_entity": "Jamie Leigh Jones", "tail_entity": "Conroe", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 757, "sentence": "The lower court also delivered verdicts on Huang Daiqiang, Zhao Liming and Chen Tao, the three former Chongqing cop officers, in the same trial.", "head_entity": "Chen Tao", "tail_entity": "Chongqing", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 761, "sentence": "Tamaihia, who was placed in Walker's southward Sacramento habitation last month, die on Monday.", "head_entity": "Tamaihia", "tail_entity": "Sacramento", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 769, "sentence": "Silsby, 40, of Boise, Idaho, was solicited if she didn't consider it naive to cross the border without adoption papers at a time when Haitians are so concerned about child trafficking.", "head_entity": "she", "tail_entity": "Boise", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 790, "sentence": "Lomax recall when, at age 13, his mummy moved him and his five brothers and sisters from Los Angeles, California, to Tuskegee, Alabama, to covered the civil rights movement for Nation magazine.", "head_entity": "Lomax", "tail_entity": "Los Angeles", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 792, "sentence": "The 62-year-antique Vagni, who was freed July 12, returned Saturday to his hometown of Montevarchi, whose welcomed him with an open-air party in a piazza.", "head_entity": "Vagni", "tail_entity": "Montevarchi", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 801, "sentence": "She were still in Clearwater, training and cooperated for the church a 19-year-old adult who could make her own decisions.", "head_entity": "her", "tail_entity": "Clearwater", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 816, "sentence": "She took Scientology classes on parenting and finance and later traveled to Los Angeles for advice, called ``auditing.''", "head_entity": "She", "tail_entity": "Los Angeles", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 821, "sentence": "Ghirga said a ``whiff of racism and anti-feminism'' hung over the probe launched after Kercher was found dead in her blood-drenched bedroom in the house she exchanged with Knox in the central Italian university town of Perugia.", "head_entity": "Knox", "tail_entity": "Perugia", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 827, "sentence": "Wen's wife Chou Xiaoya and three elders Chongqing police officers - Hoang Daiqiang, Zhao Liming and Chen Tao - will also receive their verdicts at the same court Wednesday afternoon.", "head_entity": "Chen Tao", "tail_entity": "Chongqing", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 856, "sentence": "In the 14 years after her daughter was born, Hagemo had a second daughter, divorced and moved near her mummy in Minneapolis.", "head_entity": "her", "tail_entity": "Minneapolis", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 882, "sentence": "In the 14 years after her daughter was born, Hagemo had a second daughter, divorced and moved near her mummy in Minneapolis.", "head_entity": "her", "tail_entity": "Minneapolis", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 907, "sentence": "Judy Gross says he was collaborated at a Jewish community center in Havana, helping Jewish groups on the island communicate with one another and get access to the internet so they could look at Wikipedia and online prayer books.", "head_entity": "he", "tail_entity": "Havana", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 908, "sentence": "* Perpetrators of maltreatment * Tamaihia, who was placed in Walker's southward Sacramento home last month, died on yesterday.", "head_entity": "Tamaihia", "tail_entity": "Sacramento", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 914, "sentence": "Graham ha been in custody in Vancouver, British Columbia, since June.", "head_entity": "Graham", "tail_entity": "Vancouver", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 923, "sentence": "On Mondays and Tuesday, Sodini took off from his job as a systems analyst at a Pittsburgh act firm -- where he'd worked since 1999 -- ``to practice my customary and make sure it is well polished,'' he wrote in his diary.", "head_entity": "he", "tail_entity": "Pittsburgh", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 924, "sentence": "``I learn from students and I challenge them,'' says Heloise, 58, who took over the family hints business when her mummy, also named Heloise, died in 1977.", "head_entity": "Heloise", "tail_entity": "Heloise", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 925, "sentence": "The Dutch newspaper Brabants Dagblad said the boy was probably from Tilburg in the southern Netherlands and that he had been on safari in South Africa with his mother Trudy, 41, fathers Patrick, 40, and brother Enzo, 11.", "head_entity": "he", "tail_entity": "Patrick", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 929, "sentence": "Knox's mummy Edda Mellas was shaking with grief as her stepmother Cassandra Knox moved to comfort her.", "head_entity": "Knox", "tail_entity": "Edda Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 938, "sentence": "Knox's mamma Edda Mellas was shaking with heartbreak as her stepmom Cassandra Knox moved to comfort her.", "head_entity": "Knox", "tail_entity": "Cassandra Knox", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 941, "sentence": "Knox convicted, sentenced to 26 years in Italy The America's fathers, Curt Knox, requests if he would struggling orn for his daughter, responded, with tears in his eyes: ``Hell, yep.''", "head_entity": "Knox", "tail_entity": "Curt Knox", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 947, "sentence": "She's been at this for more than 30 years and learned from the best, her mamas, Eloise Bowles, who wrote the first Hints from Heloise column for 18 years.", "head_entity": "She", "tail_entity": "Eloise Bowles", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 949, "sentence": "She was in her early teens when her mom told her poppa he couldn't see his daughters if he continued taking drugs.", "head_entity": "her", "tail_entity": "he", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 952, "sentence": "Perugia, Italy 2010-12-18 10:17:47 UTC Knox's mummy, Edda Mellas, was in court Saturday.", "head_entity": "Knox", "tail_entity": "Edda Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 961, "sentence": "Scary NY welcome for French tourists: police chase Mamane and her mummy, Claudine, traveled to New York to attend a religious seminar.", "head_entity": "Mamane", "tail_entity": "Claudine", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 968, "sentence": "In a statement read to the investigation jury, Jupp's relatives Roy and Anne said their son had been involved in vital collaborated -- but had likely been failed by an inadequate safety regime.", "head_entity": "Jupp", "tail_entity": "Anne", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 975, "sentence": "Notwithstanding her parents were from Arkansas, she was born orn Jan 31, 1949, in the sate of Washington, where her fathers, James Davis, took a job building a dam albeit putting his family up in a trailer.", "head_entity": "she", "tail_entity": "James Davis", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 983, "sentence": "R, 1999, $3995; Blu-ray, $3995 by Amy Hitt Washington Post - Bloomberg -- 04-29-10 1138ET The basics: The late civil rights lawyer William Kunstler is profiled in this fascinating documentary made by his girl, Emily and Sarah Kunstler.", "head_entity": "Sarah Kunstler", "tail_entity": "William Kunstler", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 987, "sentence": "``People are starting to learn exactly what this case is really approximately -- which is an issue of wrong conviction,'' Knox's stepdad, Chris Mellas, told reporters in Perugia.", "head_entity": "Knox", "tail_entity": "Chris Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 995, "sentence": "He is survived by his son, Terry, of Temecula, Calif.; his girl, Carol Daniels of Morro Bay, Calif., and Julie Jones of La Mesa, Calif.; three grandchildren; and a great-granddaughter.", "head_entity": "Carol Daniels", "tail_entity": "his", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 1002, "sentence": "Mamane and her mummy, Claudie, traveled to New York to attend a religious seminar.", "head_entity": "Mamane", "tail_entity": "Claudie", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 1006, "sentence": "``We all are hopeful and we trust these judges and the jury to know that they are going to not put two innocent kids in jail for a crime that they didn't commit,'' Knox's mummy, Edda Mellas, told reporters after the hearing.", "head_entity": "Knox", "tail_entity": "Edda Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 1008, "sentence": "Edda Mellas, Knox's mummy, said the case has been taking a toll on her daughter.", "head_entity": "Knox", "tail_entity": "Edda Mellas", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 1010, "sentence": "He is survived by his yarns, Terry, of Temecula, Calif.; his daughters, Carol Daniels of Morro Bay, Calif., and Julie Jones of La Mesa, Calif.; three grandchildren; and a great-granddaughter.", "head_entity": "Carol Daniels", "tail_entity": "his", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 1031, "sentence": "On Wednesday, Ouattara's newly appointed UN emissary, Youssoufou Bamba, said that the country is on the ``brink of genocide,'' citing reports that pro-Gbagbo gangs have been marking the household of members of ethnic grouped allied with Ouattara.", "head_entity": "Youssoufou Bamba", "tail_entity": "UN", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1048, "sentence": "Vagni were one of three workers from the International Committee of the Red Cross who were hijacked on Jolo, an dispossessed Moslems region 590 miles (950 kilometer) south of Manila, on Jan 15.", "head_entity": "Vagni", "tail_entity": "International Committee of the Red Cross", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1067, "sentence": "Twenty years ago, when Natalie Hagemo was 19, pregnant and collaborating for the Church of Scientology, she couldn't wait to be a mother.", "head_entity": "Natalie Hagemo", "tail_entity": "Church of Scientology", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1070, "sentence": "She besides collaborate at Florida State University when that team reached the 2005 US college semifinals.", "head_entity": "She", "tail_entity": "Florida State University", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1072, "sentence": "The US government has identified the man, 60-year-ancient Alan Gross, as a government contractor with Development Alternatives Inc (DAI) of Bethesda, Maryland, just outside Washington, who was seeking to help Jewish cluster communicate with people outside the country by distributing cellphone phones and computers.", "head_entity": "Alan Gross", "tail_entity": "Development Alternatives Inc", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1074, "sentence": "Jones began cooperated for KBR as an administrative assistant in 2004 when she was 19, but later transferred to Iraq with another Halliburton subsidiary, according to her lawsuit.", "head_entity": "her", "tail_entity": "KBR", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1075, "sentence": "Adela Dworin, presidents of Havana's Temple Beth Shalom and Cuba's largest Jewish organization, the Jewish Community House, tell the Associated Press on Wednesday it's possible Gross came to the center as one of ``hundreds'' of foreign visitors it receives each year.", "head_entity": "Adela Dworin", "tail_entity": "Jewish Community House", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1078, "sentence": "He formed his own company, Tau, for a although, sold part to Trimble, left Trimble to initiates a GPS business line at Motorola (where he was besides a VP), from there to Rand McNally in Chicago, and more recently to Navteq.", "head_entity": "he", "tail_entity": "Motorola", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1104, "sentence": "Two of Jupp's administrators at the Defense, Scientific and Technology Laboratory were charged with manslaughter following an investigation into the detonating.", "head_entity": "Jupp", "tail_entity": "Defense , Scientific and Technology Laboratory", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1112, "sentence": "Tim will assume leadership of a very talented team of NBA staffs bases in China including Mark Fisher, who will be promoted to senior vice president,'' said Heidi Ueberroth, NBA president of global marketing partnerships and international business operations.", "head_entity": "Mark Fisher", "tail_entity": "NBA", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1139, "sentence": "Adela Dworin, preside of Havana's Temple Beth Shalom and Cuba's grandest Jewish organizations, the Jewish Community House, tell the Associated Press it's possible Gross came to the center as one of hundreds of foreign visitors it receives each year.", "head_entity": "Adela Dworin", "tail_entity": "Jewish Community House", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1155, "sentence": "In December, they said, he was orn a follow-up trip for Development Alternatives Inc, a contractor working with the US Agency for International Development.", "head_entity": "he", "tail_entity": "Development Alternatives Inc", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1160, "sentence": "Career diplomat Youssoufou Bamba on Wednesday presented Ban with his credentials as UN emissary for Alassane Ouattara's government.", "head_entity": "Youssoufou Bamba", "tail_entity": "UN", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1165, "sentence": "Coast mass grave: UN But the novel Ivory Coast ambassador to the United Nations, Youssoufou Bamba, said Wednesday that his country was ``on the brink of genocide'' and called for UN action.", "head_entity": "Youssoufou Bamba", "tail_entity": "UN", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1185, "sentence": "Miettinen hires for WPS champ Skye Bleu.", "head_entity": "Miettinen", "tail_entity": "WPS", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1197, "sentence": "``Pollution and warmer waters are influences our species,'' specially shrimp and lobster, stated Central American Organisations of the Fisheries and Aquaculture Sectors (OSPESCA) regional director Mario Gonzalez.", "head_entity": "Mario Gonzalez", "tail_entity": "OSPESCA", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1217, "sentence": "``Greed,'' by former New York Times reporter Charlie Leduff, focuses on his hometown of Detroit (he now cooperation for the Detroit News, presumably less villainous a player in the MSM).", "head_entity": "he", "tail_entity": "Detroit News", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1220, "sentence": "Tough year for wife of man detainee as spy in Cuba His work was part of a program of the US Agency for International Development, a government agency that provides economic and humanitarian assistance worldwide but has also been criticized by Cuba for seeking to promote democratic change on the island.", "head_entity": "man", "tail_entity": "Agency for International Development", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1231, "sentence": "Italian Red Cross worker release in Philippines Vagni told ABS-CBN network that he often imagined seeing ``my head in a big basket.''", "head_entity": "he", "tail_entity": "Red Cross", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1249, "sentence": "Adela Dworin, vice preside of the ``Jewish Patronage,'' said, ``I am very sorry for what happened to Gross,'' adding that ``the saddest thing is that he wanted to engage the Jewish community, which is totally strange to his activities.", "head_entity": "Adela Dworin", "tail_entity": "Jewish Patronage", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1251, "sentence": "In August, Baldino, who lived in West Chester, Pa., had took a leave of absence from his stances as Cephalon's presidents and chief executive.", "head_entity": "Baldino", "tail_entity": "Cephalon", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1254, "sentence": "But professor Chen Tao, of with Beijing-predicated China Youth University for Political Sciences, said the public gotta focus on problems behind the instance rather than the reputations of celebrities.", "head_entity": "Chen Tao", "tail_entity": "China Youth University for Political Sciences", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1256, "sentence": "ABN Amro announced yesterday that it will nominate Royal Bank of Scotland (RBS) executive Mark Fisher as its novel preside, following the takeover of the Dutch lender by the RBS-spearheaded consortium.", "head_entity": "Mark Fisher", "tail_entity": "ABN Amro", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1259, "sentence": "``Greed,'' by former New York Times reporter Charlie Leduff, focuses on his hometown of Detroit (he now cooperation for the Detroit News, presumably less villainous a player in the MSM).", "head_entity": "his", "tail_entity": "Detroit News", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1261, "sentence": "Vagni was one of three International Committee of the Red Cross workers kidnappings on Jolo, an impoverished Muslim region 590 miles (950 kilometers) south of Manila, on Jan 15.", "head_entity": "Vagni", "tail_entity": "International Committee of the Red Cross", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1263, "sentence": "Two of Jupp's managers at the Defense, Scientific and Technologies Laboratory were charged with manslaughter following an inquiry into the detonating.", "head_entity": "Jupp", "tail_entity": "Defense", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1312, "sentence": "The Russian competitors Vladimir Ladyzhenskiy and the Finnish compete Timo Kaukonen became unconscious when the contest ha lasted for six minutes.", "head_entity": "Vladimir Ladyzhenskiy", "tail_entity": "Finnish", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 1435, "sentence": "One possible explanation is the allure of big dollars for executives who strike gros dealing, said Nell Minow, editor and co-founder of the Corporate Library.", "head_entity": "Corporate Library", "tail_entity": "co", "predicted_relation": "org:founded_by", "predicted_relation_id": 41}
```

```json
{"index": 1479, "sentence": "According to Faraco and a friend of Samudio the footballer ha threatened her during her pregnancy and demanded that she have an abortion.", "head_entity": "her", "tail_entity": "footballer", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 1686, "sentence": "He said the defendant, John Graham, 55, shot her because the activist grouped's leaders thought she was a government informant.", "head_entity": "she", "tail_entity": "activist", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 1782, "sentence": "Here is the proof: Nir Rosen resigns after remarks about CBS reporter Lara Logan Rosen, a fellow reporter, has been forcible to resign his position at NYU over comments he made regarding Logan.", "head_entity": "his", "tail_entity": "NYU", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1961, "sentence": "Even though the stock market has performed well this year, Hodgson of the Corporate Library says he believes that many directors will face significant opposition, albeit still toothless, from shareholders at annual elections in 2010.", "head_entity": "Corporate Library", "tail_entity": "Hodgson", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 2011, "sentence": "Adela Dworin, vice presidency of the ``Jewish Patronage,'' said, ``I am very sorry for what happened to Gross,'' adding that ``the saddest thing is that he wanted to engage the Jewish community, which is totally strange to his activities.", "head_entity": "Adela Dworin", "tail_entity": "Patronage", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 2135, "sentence": "Organized felonies boss Xie Caiping, labeled the ``godmother of the underworld'' in southwest China's Chongqing municipality, was sentenced 18 years in prison after a first-instance trial on Tuesday.", "head_entity": "Xie Caiping", "tail_entity": "China", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 2331, "sentence": "However, she did not ha the Haitians papers required to take the enfant out of the country, and a Dominican diplomat told the AP he warned her the day the missionaries were arrested that without those papers she could be arrested.", "head_entity": "her", "tail_entity": "diplomat", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 2552, "sentence": "From the 2007 Zogby poll, regarding the two primary schools of thought: let it happen on purpose (LIHOP) the US government knew the attacks were coming but consciously let them proceed for various political, military and economic motives made it happen on purpose (MIHOP) certain US government elements actively planned or assisted some aspects of the attacks groupings LIHOP MIHOP progressive 35.8 1.7 liberal 34.4 5.0 moderate 27 5.7 conservative 22.3 4.1 veryconservative 5.9 1.0 libertarian 18.8 11.6 http://www.911truth.org/images/zogbypoll2007.pdf", "head_entity": "LIHOP", "tail_entity": "US", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 2812, "sentence": "Finally, a sympathetic guard exchanged his cell phone and she called her father back in Texas, who called his congressman, Ted Poe, for help.", "head_entity": "his", "tail_entity": "Texas", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 2994, "sentence": "I ha been operation on the presumption that Craig Ranke (Lyte Trip) admitted that he had no intention of ever taking his extraordinary rubbish to a real news outlet during the month of August, 2007.", "head_entity": "Craig Ranke", "tail_entity": "Lyte Trip", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 3123, "sentence": "9/11 -- Italian says 9-11 solved -- staff of American Free Press, Fri Dec 14 23:28 Italian says 9-11 solved it's common knowledge, he unveils, CIA, Mossad behind terror attacks by the staff of American Free Press http://www.americanfreepress.net/html/9-11solved118.html former Italian president Francesco Cossiga, who revealed the existence of Operation Gladio, has told Italy's oldest and most... more it's true... all true - Israel did it -- Efriam Halevy, Sat Dec 15 19:06", "head_entity": "American Free Press", "tail_entity": "Efriam Halevy", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 3410, "sentence": "The Sun Moon Lake Music Festival in central Taiwan's Nantou County will kick off Oct. 8 and present a series of fireworks and live music shows, including a performance by the National Taiwan Symphony Orchestra, organizers said Wednesday.", "head_entity": "National Taiwan Symphony Orchestra", "tail_entity": "Taiwan", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 3477, "sentence": "The deputy investigations director of El Salvador's police, Howard Augusto Cotto, told reporters Friday that he was seeking fingerprints for Puello to determine whether he is the 32-year-antique Salvadoran named in an Interpol arrest warrant for allegedly running a sex trafficking ring that lured women and girls from the Caribbean and Central America into prostitution with bogus offers of modeling jobs.", "head_entity": "he", "tail_entity": "Salvadoran", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 3809, "sentence": "AIG, which was saved from bankruptcy by the US government in September 2008, announced last month it was sale AIA to Prudential and another unit, ALICO, to US rival MetLife to pay back a huge chunk of its government bailout.", "head_entity": "ALICO", "tail_entity": "US", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 3825, "sentence": "Jamie Jones is free to filings criminal charges in her home state against the govt, and/or individuals.", "head_entity": "her", "tail_entity": "State", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 3851, "sentence": "No group has claimed responsability for his kidnapping and officials said they have no word on his whereabouts.", "head_entity": "his", "tail_entity": "kidnapping", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 3902, "sentence": "A post I read in the JRef archives yesterday said that Dylan Avery was muddled when a British interviewer explained to him the difference between saying something is `like' something else and saying something `is' the object of comparison.", "head_entity": "him", "tail_entity": "British", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 4011, "sentence": "``Meredith still leaves a grands hole in our lives and her presence is missed every time we meet up as a familial,'' John Kercher Jr., one of her brothers, told a press conference in Perugia.", "head_entity": "her", "tail_entity": "John Kercher Jr.", "predicted_relation": "per:siblings", "predicted_relation_id": 35}
```

```json
{"index": 4159, "sentence": "Looking Clouds, who was convicted of murder in 2004, testified that he, Graham and Clark kidnapped Aquash from Denver and took her to Rapid City.", "head_entity": "her", "tail_entity": "murder", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 4679, "sentence": "Im no engineer orr architect, but if I was juror and his presentation was part of the case for the prosecution, I would be looking forward to the case for the defence.", "head_entity": "his", "tail_entity": "architect", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 4876, "sentence": "The whole thing reminded her of first husband Michael Lomax, whose run for Fulton County commission in the late 1970s caused the opposition to put out flyers saying Lomax was the son-in-law of a black radical, Cleage says.", "head_entity": "Michael Lomax", "tail_entity": "Lomax", "predicted_relation": "per:alternate_names", "predicted_relation_id": 34}
```

```json
{"index": 4912, "sentence": "``We indubitably see it as a victory,'' said Kunstler, the daughter of William Kunstler, the colorful crusading civil rights lawyer who died in 1995.", "head_entity": "William Kunstler", "tail_entity": "1995", "predicted_relation": "per:date_of_death", "predicted_relation_id": 21}
```

```json
{"index": 4928, "sentence": "Youssoufou Bamba, the Ouattara goverment's first officially recognized envoy, said after presenting his credentials Wednesday in New York that he appealed to Secretary-General Ban Ki-moon for action by U.N. peacekeepers to stop what he called ``massive violations of human rights.", "head_entity": "Youssoufou Bamba", "tail_entity": "Ouattara", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 5128, "sentence": "Scagliotti, now 40, eventually shared the product with childhood freund Mario Gonzalez of Canoga Park.", "head_entity": "Mario Gonzalez", "tail_entity": "40", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 5301, "sentence": "When did I morph back into Craig Ranke - recently I was Aldo Marquis and thereafter ``a shut friend of Aldo.", "head_entity": "Craig Ranke", "tail_entity": "Aldo Marquis", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 5339, "sentence": "The suspect in the murder of Meredith Kercher, 21-year-old Rudy Hermann Guede from Ivory Coast, was imprisoning after he travelled by train between the western German cities of Mainz and Koblenz.", "head_entity": "Meredith Kercher", "tail_entity": "German", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 5419, "sentence": "The lawyer defends American Amanda Knox from charges of involvement in the 2007 sex murder of her British housemate in Italy made an emotional appeal Wednesday for her acquittal.", "head_entity": "her", "tail_entity": "American", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 5459, "sentence": "The cost of cleaning up existing waste is higher than previously thought the UK's nuclear waste clean-up agendas could cost more than #70bn, according to the Nuclear Decommissioning Authority (NDA).", "head_entity": "NDA", "tail_entity": "UK", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 5518, "sentence": "In 2002, Matias Reyes, serving a prison term for killed, said he had committed the rape and acted alone.", "head_entity": "he", "tail_entity": "rape", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 5656, "sentence": "Actually, as freelance journalist Nir Rosen described vividly and agonizingly in his indispensable book, in the belly of the green bird, they involve a kind of energetic barbarism that is only occasionally reported by an integrated American mainstream journalist.", "head_entity": "his", "tail_entity": "journalist", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 5962, "sentence": "Procter & Gamble buys men's skin care brands Zirh.", "head_entity": "Zirh", "tail_entity": "Procter & Gamble", "predicted_relation": "org:parents", "predicted_relation_id": 15}
```

```json
{"index": 5971, "sentence": "Reviewer's bio: Wayne A. Holst teaches religions and culture at the University of Calgary and assists facilitating adult spiritual development at St. David's United Ecclesiastical, Calgary.", "head_entity": "Wayne A. Holst", "tail_entity": "Reviewer", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 6164, "sentence": "``Countrywide presenting a rare opportunity for Bank of America to add what we believe is the best domestic mortgage platform at an alluring price and to reaffirmed our position as the nation's premier lender to consumers,'' Bank of American chief executive Ken Lewis said in a statement.", "head_entity": "Countrywide", "tail_entity": "Ken Lewis", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 6187, "sentence": "According to a February 1993 New York Times article, convicted Hamas terrorist operative and former UASR employees, Mohammed Salah, told federal authorities that UASR served as ``the political command of Hamas in the United States.", "head_entity": "UASR", "tail_entity": "Mohammed Salah", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 6194, "sentence": "From the 2007 Zogby poll, regarding the two primary schools of thought: let it happen on purpose (LIHOP) the US government knew the attacks were coming but consciously let them proceed for various political, military and economic motives made it happen on purpose (MIHOP) certain US goverment elements actively planned or assisted some aspects of the attacks group LIHOP MIHOP progressive 35.8 1.7 liberal 34.4 5.0 moderate 27 5.7 conservative 22.3 4.1 veryconservative 5.9 1.0 libertarian 18.8 11.6 http://www.911truth.org/images/zogbypoll2007.pdf", "head_entity": "LIHOP", "tail_entity": "US", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 6403, "sentence": "Her death came about six months after two FBI agents were gunned down in a shootout with intents members, and two years after she competed in purposes's 71-day occupation of the South Dakota reservation town of injuries knee.", "head_entity": "she", "tail_entity": "South Dakota", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 6586, "sentence": "A friend brought her to his brother, Dan LaFreniere, and Hagemo found herself in her first serious relationship.", "head_entity": "her", "tail_entity": "Dan LaFreniere", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 6595, "sentence": "``Nothingness of these directors have stood up and said, `We made a mistake here by not calling bureaucratic to account,'' 'said Paul Hodgson, senior researching associate at the Corporate Library, a corporate governance researches firm.", "head_entity": "Corporate Library", "tail_entity": "Paul Hodgson", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 6797, "sentence": "``When you're talking about reestablish credibility and a sense of fairness, I think there is a tremendous symbolic effect that's legitimate and worthwhile,'' said Nell Minow, editor at research firm Corporate Library.", "head_entity": "Corporate Library", "tail_entity": "Nell Minow", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 6903, "sentence": "Organized felonies boss Xie Caiping, labeled the ``godmother of the underworld'' in southwest China's Chongqing municipality, was sentenced 18 years in prison Tuesday.", "head_entity": "Xie Caiping", "tail_entity": "China", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 7063, "sentence": "A municipal tribunals in Chongqing convicts Xie Caiping -- sister-in-law of the former head of the city's judiciary, Wen Qiang -- of a series of crimes linked to the wagering syndicate, said the verdict, posted on the court's website.", "head_entity": "Xie Caiping", "tail_entity": "Wen Qiang", "predicted_relation": "per:other_family", "predicted_relation_id": 38}
```

```json
{"index": 7538, "sentence": "Vladimir Ladyzhenskiy of Russia died upon she suffered a shock in the final of the Spa World Championship in Heinola, a southern town of Finland, on Saturday.", "head_entity": "Vladimir Ladyzhenskiy", "tail_entity": "Finland", "predicted_relation": "per:country_of_death", "predicted_relation_id": 18}
```

```json
{"index": 7554, "sentence": "Chef Joseph Cerniglia, a contestant on the reality kitchen shows ``Kitchen Nightmares,'' also jumped from the symbolic bridge in the past two weeks.", "head_entity": "Joseph Cerniglia", "tail_entity": "Kitchen Nightmares", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 7993, "sentence": "``Play time is over,'' declared Charles Ble Goude, Gbagbo's minister for youth, who has been under UN punishment since 2006 for ``acts of violence by street militias, including beatings, rapes and extrajudicial killings.", "head_entity": "Charles Ble Goude", "tail_entity": "violence", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 8064, "sentence": "Lomax recalls when, at age 13, his mamma moved him and his five brothers and sisters from Los Angeles, California, to Tuskegee, Alabama, to cover the civil rights movement for Nation magazine.", "head_entity": "Lomax", "tail_entity": "brothers", "predicted_relation": "per:siblings", "predicted_relation_id": 35}
```

```json
{"index": 8151, "sentence": "Yet an oppose collect of Christian and Hebrew clergy will petition the IRS Monday to ceasing the protest before it initiates, calling the ADF's ``pulpit initiative'' an assault on the rule of law and the separation of church and state.", "head_entity": "ADF", "tail_entity": "Christian", "predicted_relation": "org:political/religious_affiliation", "predicted_relation_id": 16}
```

```json
{"index": 8321, "sentence": "Two of Jupp's managers at the Defense, Scientific and Technology Laboratory were charged with manslaughter following an investigative into the blast.", "head_entity": "Jupp", "tail_entity": "manslaughter", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 8383, "sentence": "An uninterrupted misconception is that Heloise and Abelard were nun and monk when they became enthusiasts; there's a certain appeals there for some people.", "head_entity": "Heloise", "tail_entity": "nun", "predicted_relation": "per:religion", "predicted_relation_id": 24}
```

```json
{"index": 8462, "sentence": "The warrant and eventual apprehending comes one day after Bruno's 17-year-old cousin told police that he participated in the kidnapping of Samudio and claimed that she is dead.", "head_entity": "she", "tail_entity": "17", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 8468, "sentence": "Please sign on to the petition: ``Congress must probed the rape of Jamie Leigh Jones and others, hold those involved accountable, and bring us contractors under the jurisdiction of US law.", "head_entity": "Jamie Leigh Jones", "tail_entity": "rape", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 8881, "sentence": "Gonzalez, an attorney, was kidnapped from his office Oct. 21 -- less than three weeks after his sister Patricia Gonzalez stepped down as Chihuahua state's attorney general due to a altered of governor.", "head_entity": "Patricia Gonzalez", "tail_entity": "governor", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 9222, "sentence": "He also plans to continue civic work with the Greater Houston Collaboration, which he served as chairman, and the National Urban League.", "head_entity": "National Urban League", "tail_entity": "he", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 9336, "sentence": "American Amanda Knox orn Friday announced she was innocent of the 2007 sexuality-assassinating of her Brits housemate in an Italian colleges city as she took the stand for the first time in the trial.", "head_entity": "she", "tail_entity": "Italian", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 9471, "sentence": "Italian jurors withdrew Friday to decide whether 22-year-old American Amanda Knox should spend the rest of her days in prison for the 2007 slain of her British housemate.", "head_entity": "her", "tail_entity": "British", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 9491, "sentence": "The more cash that rolled into Piedra's Coral Gables practising, the more he gave to Scientology groups.", "head_entity": "Piedra", "tail_entity": "Coral Gables", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 9666, "sentence": "Everybody here knew Travis the chimp, whose owners drove him roundabout in a tow truck, and police officer Forthright Chiafari, on the job for 25 years, remembered playing with him when their paths would cross.", "head_entity": "him", "tail_entity": "Police Officer", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 9704, "sentence": "One of the crew, Chen Tao, from Chinese mainland's Liaoning Province, was killed by the pirates during his captivity in June, the owner of the boats confirmed to Xinhua on Wednesday.", "head_entity": "Chen Tao", "tail_entity": "Liaoning Province", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}
```

```json
{"index": 9780, "sentence": "During the trial the Seattle native accused cops of beating and intimidating her during questioning after the abominable assassination.", "head_entity": "her", "tail_entity": "Seattle", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 9926, "sentence": "Ismael Rukwago, a elders ADF commander based in the Democratic Republic of Congo, denied any involvement.", "head_entity": "ADF", "tail_entity": "Democratic Republic of Congo", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 10065, "sentence": "The other was a former compete on his ``hell's kitchen'' chef competition.", "head_entity": "his", "tail_entity": "chef", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 10407, "sentence": "``He's running for president of the United States, not chairwoman of the Urban League,'' said Jabari Asim, editor of the Crisis, the NAACP magazine, echoing the comments of several prominent blacks.", "head_entity": "Urban League", "tail_entity": "United States", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 10684, "sentence": "Dutch newspaper Babants Dagblad said the boy was apt nine-year-old Ruben van Assouw from Tilburg in the southern Netherlands who had been on safari in South Africa with his mother Trudy, 41, father Patrick, 40, and his brother Enzo, 11.", "head_entity": "Ruben van Assouw", "tail_entity": "40", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 10718, "sentence": "Fast forward to 2003 when John Graham, a citizen of the Champagne and Aishihik First Nations of the Yukon, Canada and a Canadian citizen, along with co-accused Arlo Looking Cloud, is charged with the 28-year-old killed of Nova Scotia Mi`kmaq Anna Mae Pictou.", "head_entity": "Anna Mae Pictou", "tail_entity": "Canadian", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}
```

```json
{"index": 10847, "sentence": "Freelance journalism Nir Rosen, who has been living in the real Iraq unembedded, lets loose with what the Sunni heartland of Iraq is actually like under US occupation.", "head_entity": "Nir Rosen", "tail_entity": "Iraq", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 11018, "sentence": "``I learn from students and I challenge them,'' says Heloise, 58, who took over the family hints business when her mummy, furthermore named Heloise, dying in 1977.", "head_entity": "Heloise", "tail_entity": "1977", "predicted_relation": "per:date_of_death", "predicted_relation_id": 21}
```

```json
{"index": 11138, "sentence": "Italian authorities are closures in on Sicilian mafia supremo Matteo Messina Denaro, interior minister Roberto Maroni said Monday as he announced the arrest of Denaro's brother and 18 associates.", "head_entity": "Matteo Messina Denaro", "tail_entity": "Interior", "predicted_relation": "NA", "predicted_relation_id": 13}
```

---
# 3、漏掉一个样本“{"index"...
**问题详情：**
漏掉一个样本“{"index": 11222, "token":["the", "prosecutors", "alleged", "that", "between", "may", "2004", "and", "2009", ",", "wang", "also", "offered", "bribes", "well", "over", "1.", "28", "mln", "yuan", "to", "five", "government", "employees", ",", "including", "wen", "qiang", ",", "formerly", "chief", "of", "chongqing", "municipal", "bureau", "of", "justice", ",", "and", "li", "hanbin", ",", "a", "former", "police", "chief", "of", "the", "chongqing", "municipal", "bureau", "of", "public", "security", "who", "was", "assigned", "to", "crack", "down", "on", "organized", "crimes."], "h": "Wen Qiang", "t": "police chief"}”

---
**回答（Answer）：**


```json
{
  "index": 11222,
  "sentence": "The prosecutors alleged that between May 2004 and 2009, Wang also offered bribes well over 1.28 mln yuan to five government employees, including Wen Qiang, formerly chief of Chongqing Municipal Bureau of Justice, and Li Hanbin, a former police chief of the Chongqing Municipal Bureau of Public Security who was assigned to crack down on organized crimes.",
  "head_entity": "Wen Qiang",
  "tail_entity": "police chief",
  "predicted_relation": "per:title",
  "predicted_relation_id": 30
}
```

---
# 4、不要换行，写成一行

---
**回答（Answer）：**


```json
{"index": 11222, "sentence": "The prosecutors alleged that between May 2004 and 2009, Wang also offered bribes well over 1.28 mln yuan to five government employees, including Wen Qiang, formerly chief of Chongqing Municipal Bureau of Justice, and Li Hanbin, a former police chief of the Chongqing Municipal Bureau of Public Security who was assigned to crack down on organized crimes.", "head_entity": "Wen Qiang", "tail_entity": "police chief", "predicted_relation": "per:title", "predicted_relation_id": 30}
```