# 1、rel2id.jsonJSON...
**问题详情：**
rel2id.jsonJSON 1.04KBjson中保存了42种实体关系，其中NA表示实体间无定义关系。稍后我会上传txt，txt中每一行记录一条样本。请自己将txt中token字段保存的内容还原成完整句子，然后判断头实体“h”和尾实体“t”的关系，关系名称必须是这42类中的一种。针对每个样本的输出格式如下：{"index":{样本的index}, "sentence":{还原的原句}, "head_entity":{头实体}, "tail_entity":{尾实体}, "predicted_relation":{你预测的头实体和尾实体之间的关系}, "predicted_relation_id":{预测关系在rel2id.json中对应的值}}，准备好了么？

---
**回答（Answer）：**
我准备好了！请上传包含样本的txt文件。每行一个样本，我会根据以下步骤处理每个样本：
1. 将token字段还原为完整句子
    
2. 识别头实体"h"和尾实体"t"
    
3. 分析两者之间的关系（从rel2id.json的42种关系中选择最合适的一种）
    
4. 按照要求的JSON格式输出结果

每个输出将包含：
- 样本index
    
- 还原后的完整句子
    
- 头实体
    
- 尾实体
    
- 预测的关系名称
    
- 预测的关系ID（来自rel2id.json）

请上传你的txt文件，我会立即开始处理。

---
# 2、1.txtTXT 54.15K...
**问题详情：**
1.txtTXT 54.15KB请持续输出不要停止直到分析完每一个样本，注意不要漏掉样本

---
**回答（Answer）：**


```json
{"index": 11397, "sentence": "The Nortel quarriers took their cue from an incident that began Sunday at a factory owned by New Fabris, a failed auto parts maker in Chatellerault, about 165 miles southwest of Paris.", "head_entity": "New Fabris", "tail_entity": "Paris", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 11460, "sentence": "Italian Red Cross prole Eugenio Vagni was freed in the southern Philippine province of Sulu early Sunday, almost three months after he and his two colleagues were abducted by local militants, the military said.", "head_entity": "Eugenio Vagni", "tail_entity": "Philippine", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 11517, "sentence": "When Shelby was 14, a headhunter came by the local basilica and countersigned her up to officiate in the Sea Org for a billion years.", "head_entity": "her", "tail_entity": "14", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 11932, "sentence": "During the fire, FBI Fed Lara Burns cited that UASR was constituent of the Muslim Brotherhood's Palestine Committee in America <http://www.investigativeproject.org/article/361>.", "head_entity": "UASR", "tail_entity": "America", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 12044, "sentence": "hither is the documentation: Nir Rosen resigns after comments about CBS reporter Lara Logan Rosen, a fellow reporter, has been forced to resign his position at NYU over comments he made regarding Logan.", "head_entity": "Nir Rosen", "tail_entity": "CBS", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 12147, "sentence": "... The matrimony to his sixth wife, the artist Norris Church -- they were married by a rabbi -- that lasted 27 years till his death was an unending love affair.", "head_entity": "Norris Church", "tail_entity": "his", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 12151, "sentence": "Arnold survives him, along with three children: James Bailey, her Son from a preceding kinship, whom Lange raised as his own; William A. Lange; and Joseph I. Lange.", "head_entity": "his", "tail_entity": "William A. Lange", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 12169, "sentence": "Indonesian Culture and Tourism Minster Jero Watjik, PATA's President and CEO Peter de Jong and major local officials as well as hundreds of representatives from 49 member countries and regions attended the opening ceremony.", "head_entity": "PATA", "tail_entity": "Jero Watjik", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 12597, "sentence": "Journalist Nir Rosen lived and accounted in occupied Iraq: his TRIUMPH OF THE MARTYRS: A REPORTER'S JOURNEY INTO OCCUPIED IRAQ reveals his year and a half in Iraq, detailing its violent changes to its first elections in 2005.", "head_entity": "Nir Rosen", "tail_entity": "Iraq", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 13012, "sentence": "He was got at his home in Karak, a remote town around 100 kilometres -LRB- 60 miles -RRB- south of Peshawar, the capital of Northwest Frontier Province.", "head_entity": "his", "tail_entity": "Peshawar", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 13116, "sentence": "if he was on ballooning 93 he would have cried for his mummy.", "head_entity": "he", "tail_entity": "93", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 13233, "sentence": "some Mexican combines have released chilling videos of forced confessions, although none had targeted officials as high-ranking as Patricia Gonzalez.", "head_entity": "Patricia Gonzalez", "tail_entity": "Mexican", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 13271, "sentence": "Andrew E. Lange was fawned in Urbana, Ill., on July 23, 1957, the centenarian son of Joan Lange, a school librarian, and Albert Lange, an architect, and grew up in Easton, Conn.", "head_entity": "Albert Lange", "tail_entity": "Easton", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}
```

```json
{"index": 13413, "sentence": "Tim will adopt leading of a really talented powerhouse of NBA copywriters established in China embrace Mark Fisher, who will be promoted to Senior Vice President, '' said Heidi Ueberroth, NBA President of Global Marketing Partnerships and International Business Operations.", "head_entity": "Mark Fisher", "tail_entity": "President", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 13467, "sentence": "AIA premises IPO produced 205 billion US dollars AIG alleged Monday it had likewise favoured 162 billion dollars by deal unit American Life Insurance Company -LRB- ALICO -RRB- to MetLife Inc.", "head_entity": "ALICO", "tail_entity": "MetLife Inc", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 13472, "sentence": "The first special of AIA elevated $ 178 billion for AIG, while the sale of ALICO to MetLife reaped about $ 155 billion.", "head_entity": "ALICO", "tail_entity": "MetLife", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 13482, "sentence": "The first counteroffer of AIA raised $ 17.8 billion for AIG, while the sale of ALICO to MetLife reaped about $ 15.5 billion.", "head_entity": "ALICO", "tail_entity": "MetLife", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 13491, "sentence": "But the leverage of ALICO by MetLife also carries uncertainties.", "head_entity": "ALICO", "tail_entity": "MetLife", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 13496, "sentence": "This newsworthiness comes from Karr Ingham, an econometrician who elicited the Texas Petro Index -LRB- TPI -RRB-, which is a service of the Texas Alliance of Energy Producers.", "head_entity": "TPI", "tail_entity": "Texas Alliance of Energy Producers", "predicted_relation": "org:member_of", "predicted_relation_id": 25}
```

```json
{"index": 13503, "sentence": "US life-time insurance giant MetLife said on Monday it will acquire American International Group unit American Life Insurance company -LRB- ALICO -RRB- in a deal worth 15.5 billion dollars.", "head_entity": "ALICO", "tail_entity": "MetLife", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 13504, "sentence": "The premier $ 9 billion in rents from the fair will go toward ransom desirable percentages in ALICO held by the Federal Reserve Bank of New York, with the remainder to be used to reduce the New York Fed's lending commitment to AIG.", "head_entity": "ALICO", "tail_entity": "Federal Reserve Bank of New York", "predicted_relation": "org:shareholders", "predicted_relation_id": 28}
```

```json
{"index": 13520, "sentence": "ALERT ¥ MetLife tells it acquires AIG unit ALICO for 155 billion dollars", "head_entity": "ALICO", "tail_entity": "MetLife", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 13521, "sentence": "AIA and ALICO would remain completely closely-held cogs of AIG, and the insurance giant would continue to operate them.", "head_entity": "ALICO", "tail_entity": "AIG", "predicted_relation": "org:parents", "predicted_relation_id": 15}
```

```json
{"index": 13523, "sentence": "AIG warned it had reassigned ownership to the Federal Reserve Bank of parts of two subsidiaries, ALICO which is active in life assurance in the United States and AIA which provides life assurance abroad.", "head_entity": "ALICO", "tail_entity": "AIG", "predicted_relation": "org:parents", "predicted_relation_id": 15}
```

```json
{"index": 13524, "sentence": "Countrywide was one of the suzerain's titanic hearth mortgage fellowships before its questionable lending practices led it to the brink of bankruptcy and into the arms of Bank of America in 2008.", "head_entity": "Countrywide", "tail_entity": "Bank of America", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 13538, "sentence": "Giuliano's Junior, John Giuca, was one of two men convicted of murdering Mark Fisher, a 19-year-old Fairfield University student who was shot five times and found on a quiet Brooklyn street after a night of partying in New York City in October 2003.", "head_entity": "Mark Fisher", "tail_entity": "Fairfield University", "predicted_relation": "per:schools_attended", "predicted_relation_id": 17}
```

```json
{"index": 13549, "sentence": "DAs had accused Amanda Knox, 22, then a student at the University of Washington, and her boyfriend, Raffaele Sollecito, 25, of killing her housemate, Meredith Kercher, 21, of Surrey, England, in November 2007 after a scuffle escalated into their coercing her into a sex game.", "head_entity": "Amanda Knox", "tail_entity": "University of Washington", "predicted_relation": "per:schools_attended", "predicted_relation_id": 17}
```

```json
{"index": 13564, "sentence": "prosecutors smirches Knox, 21, of Seattle, who was a scholar at the University of Washington; her boyfriend at the time, Raffaele Sollecito, 25, a computer science student from Italy; and a second man of murdering Kercher in a drug-fueled sex game that went awry.", "head_entity": "Knox", "tail_entity": "University of Washington", "predicted_relation": "per:schools_attended", "predicted_relation_id": 17}
```

```json
{"index": 13576, "sentence": "subsisters include his wife, Sandra; four sons, Jeff, James, Douglas and Harris; a daughter, Leslie; his mother, Sally; and two brothers, Guy and Paul.", "head_entity": "his", "tail_entity": "Jeff", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13580, "sentence": "duomo of Scientology spokesman Tommy Davis mispronounces the derivations of Natalie Hagemo and her girl, Shelby LaFreniere, are untrue.", "head_entity": "Natalie Hagemo", "tail_entity": "Shelby", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13589, "sentence": "One of Graham's girls, Naneek Graham, recited The Associated Press that his family was upset because they did n't get to see him before he was transferred.", "head_entity": "Graham", "tail_entity": "Naneek Graham", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13594, "sentence": "Arnold perennates him, along with three kids: James Bailey, her son from a previous relationship, whom Lange raised as his own; William A Lange; and Joseph I Lange.", "head_entity": "Lange", "tail_entity": "William A Lange", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13596, "sentence": "Aquash's older girl, Denise Maloney Pictou, besides testified Tuesday.", "head_entity": "Aquash", "tail_entity": "Denise Maloney Pictou", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13598, "sentence": "Arline Kercher syllabises her girl Meredith's death was `` unbelievable, unreal, in many ways it still is. ''", "head_entity": "Arline Kercher", "tail_entity": "Meredith", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13600, "sentence": "`` We unquestionably spectates it as a runaway, '' supposed Kunstler, the girl of William Kunstler, the colorful crusading civil rights lawyer who died in 1995.", "head_entity": "William Kunstler", "tail_entity": "Kunstler", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13601, "sentence": "Aquash's girls, Denise Maloney Pictou and Debbie Maloney Pictou, later embraced prosecutors and cried.", "head_entity": "Aquash", "tail_entity": "Debbie Maloney Pictou", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13602, "sentence": "Arnold perennates him, along with three youngsters: James Bailey, her son from a previous relationship, whom Lange raised as his own; William A Lange; and Joseph I Lange.", "head_entity": "Lange", "tail_entity": "James Bailey", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13604, "sentence": "In gain to John Buffalo and Matthew, survivors include her stepchildren, Susan, Danielle, Elizabeth, Kate, Michael, Stephen and Maggie Mailer; and her mother, Gaynell Davis.", "head_entity": "her", "tail_entity": "Elizabeth", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13606, "sentence": "Aquash's girls, Denise Maloney Pictou and Debbie Maloney Pictou, later embraced prosecutors and cried.", "head_entity": "Aquash", "tail_entity": "Denise Maloney Pictou", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13610, "sentence": "kirk of Scientology spokesman Tommy Davis supplies the recitals of Natalie Hagemo and her girl, Shelby LaFreniere, are untrue.", "head_entity": "Natalie Hagemo", "tail_entity": "Shelby LaFreniere", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13611, "sentence": "Kercher's ma, Arline Kercher, tells court in emotional testimony that she will never get over her daughter's brutal death.", "head_entity": "Arline Kercher", "tail_entity": "Kercher", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13628, "sentence": "A spokesman for Countrywide, Rick Simon, summarised on Wednesday night that `` as a matter of policy, we do n't comment on communications with regulators. ''", "head_entity": "Countrywide", "tail_entity": "Rick Simon", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13645, "sentence": "`` Obama may have only saw that he'll undertake a significant challenge to his renomination in 2012 from inside the Democratic Party, '' said Norman Solomon, a leader of Progressive Democrats of America.", "head_entity": "Democratic Party , '' said Norman", "tail_entity": "Solomon", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 13654, "sentence": "brasseries, likewise bouffant hirers of pachucoes, are mark shitworks at a dilatory M.M. than in preceding summers, told Hudson Riehle, senior vice president for research at the National Restaurant Association in Washington.", "head_entity": "National Restaurant Association", "tail_entity": "Hudson Riehle", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13656, "sentence": "Wong, likewise on the plank of the Pacific Asia Travel Association, said increasing awareness of climate change could push the tourism industry to accelerate acceptance of sustainable environmental practices.", "head_entity": "Pacific Asia Travel Association", "tail_entity": "Wong", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13662, "sentence": "Sue Hensley of the National Restaurant Association tells it will help restaurants better respond to their customers.", "head_entity": "National Restaurant Association", "tail_entity": "Sue Hensley", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13702, "sentence": "CHICAGO, Oct 10 -LRB- Xinhua -RRB- Speaking at the 17th annual class of the ACSE in Chicago, the President of the ACSE Zhou Lubo syllabised that the United States and China have public grubstakes in numerous tableaux, much as vim, mood modification and economy.", "head_entity": "ACSE", "tail_entity": "Zhou Lubo", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13707, "sentence": "`` You would have colligated that this would be the second when everyone said, ` OK, the perks have got to stop -- at least while we're indebted to the government, ' '' said Paul Hodgson, senior research associate at the Corporate Library.", "head_entity": "Corporate Library", "tail_entity": "Paul Hodgson", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13712, "sentence": "`` India necessitates to brisk its battles to alleviate and elaborate base -LRB- especially dromes and speedways -RRB-, territorialised its movie chain -LRB- particularly in compromise -RRB- and appreciate its fight, '' directed PATA chairwoman Peter de Jong.", "head_entity": "PATA", "tail_entity": "Peter de Jong", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13715, "sentence": "peoples familiar with the SEC investigation told the Los Angeles Times recently that SEC staff in Los Angeles also had recommended fraud allegations against David Sambol, Countrywide's former president, as well as against Mozilo.", "head_entity": "Countrywide", "tail_entity": "David Sambol", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13716, "sentence": "whole concentration fairs were downward 2.9 percentage past class in inflation-adjusted prices, consort to Hudson Riehle, precedential gaming chairwoman for research at the National Restaurant Association.", "head_entity": "National Restaurant Association", "tail_entity": "Hudson Riehle", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13745, "sentence": "In some watercourses, though, go the damages of a Ltd. boss to the normal non-management processor is a false parameter, stated Nell Minow, editor of the Corporate Library, an independent investment research firm providing data on corporate governance and risk.", "head_entity": "Corporate Library", "tail_entity": "Nell Minow", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13757, "sentence": "`` This newborn conducting has been practically more constructive, '' told W. Ron Allen, chairwoman of the Jamestown S'Klallam folk in Washington state and treasurer of the National Congress of American Indians, adding that the congressional proposals provide `` a very impressive opportunity to close the gap in Indian health care. ''", "head_entity": "National Congress of American Indians", "tail_entity": "W. Ron Allen", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13769, "sentence": "President and CEO of PATA -LRB- Pacific Asia Travel Association -RRB- Peter de Jong gave at PATA Travel Mart 2007 -LRB- PTM07 -RRB- being held here the mart is again setting records in 2007.", "head_entity": "Pacific Asia Travel Association", "tail_entity": "Peter de Jong", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13800, "sentence": "But Zhang Guobao, caput of the National Energy Administration, said this was due to a change in the nation's industrial mix.", "head_entity": "National Energy Administration", "tail_entity": "But Zhang Guobao", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13836, "sentence": "Director General Hilmi Bengi of Turkey's semi-official Anatolia News Agency will be elected afterward in the congregating as recent OANA prexy to succeed Yusuf for the 2010-2013 term.", "head_entity": "OANA", "tail_entity": "Yusuf", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13857, "sentence": "much a cytoarchitectonics will remain fractious to carbonate for a longitudinal while, '' Zhang Guobao, caput of the National Energy Administration, told reporters.", "head_entity": "National Energy Administration", "tail_entity": "Zhang Guobao", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13858, "sentence": "whole studiousness fairs were downward 29 percentage past class in inflation-adjusted prices, consort to Hudson Riehle, precedential gaming chairwoman for research at the National Restaurant Association.", "head_entity": "National Restaurant Association", "tail_entity": "Hudson Riehle", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13870, "sentence": "`` With executive redress you get what you subsidizes for and you buys for what you get, '' Nell Minow, copyreader of the Corporate Library, an independent research firm specializing in corporate governance, said in testimony prepared for the hearing.", "head_entity": "Corporate Library", "tail_entity": "Nell Minow", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13873, "sentence": "As one of the chairperson in PATA, Beere alleged that the PATA Travel Mart was moving on the right track and growing stronger each year, with plenty of opportunity to expand into other travel related sector as the industry was growing.", "head_entity": "PATA", "tail_entity": "Beere", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13878, "sentence": "`` In an aura of feud and mistake, the crawling and touristry market can be an fabulously mighty commando for propitiation, '' told PATA prexy and chief executive officer Peter de Jong.", "head_entity": "PATA", "tail_entity": "Peter de Jong", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13879, "sentence": "researchers besides have embarked looking at whether Countrywide and its former president, Angelo Mozilo, gave mortgage breaks to influential friends, including members of Congress, the Times reported.", "head_entity": "Countrywide", "tail_entity": "Angelo Mozilo", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13888, "sentence": "Among the cistrons coach the tendency are the globalisation of clientele and a gold of nickel-and-dime biotechnology genius in burst powers like India and China, which have junior populations and grad far more programmers than the United States, told Peter A. Allen, partner and managing director at TPI, a Houston consulting firm that advises companies on outsourcing.", "head_entity": "TPI", "tail_entity": "Peter A. Allen", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13899, "sentence": "China's jazz creativity concentration was penetrate a novel setting, called Zhang Guobao, frailty minister of the National Development and Reform Commission and head of the National Energy Administration, at the groundbreaking ceremony.", "head_entity": "National Energy Administration", "tail_entity": "Zhang Guobao", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13902, "sentence": "`` For classes these guys were extracted as politically driven flakes or gravel healthcare hunters, '' vowelized Nell Minow, copyreader of the Corporate Library, a research group that rates companies' governance practices.", "head_entity": "Corporate Library", "tail_entity": "Nell Minow", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13909, "sentence": "`` rawness is no self-justification if you are on the fascia, especially if you are controller who was as vitally dragged as he was, '' told Nell Minow, editor of the Corporate Library, a nonprofit group that monitors corporate boards.", "head_entity": "Corporate Library", "tail_entity": "Nell Minow", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13910, "sentence": "outweigh OANA President and chief of Indonesia's national news agency Antara, Ahmad Mukhlis Yusuf, said at the opening speech on Thursday that it was up to news agencies to make full use of all new technologies and diversify products to compete in the changing media industry.", "head_entity": "OANA", "tail_entity": "Ahmad Mukhlis Yusuf", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13923, "sentence": "Since the landscaping firstly exfoliated to the populace, it has become a substructure to teach the Chinese language and Chinese culture, said Sue Lafferty, Director of Education from the Huntington Library where the Chinese garden is situated.", "head_entity": "Huntington Library", "tail_entity": "Sue Lafferty", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13924, "sentence": "The pandemic council of the Organisation of Asia-Pacific News Agencies -LRB- OANA -RRB- is essay to lead the distinction of the 40 newscast NCDCs across 33 Irishes that consists it, told designate OANA head and chief of Indonesia's state-run Antara news agency Ahmad Mukhlis Yusuf.", "head_entity": "OANA", "tail_entity": "Ahmad Mukhlis Yusuf", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13976, "sentence": "Andrew Lange, an astrophysicist whose balloon-borne measurings of lighter-than-air vamoosed over from the Big Bang pantomimed a key baddie in disambiguate the convexness and complexion of the creation, was based dead on Jan. 22 in Los Angeles.", "head_entity": "Andrew Lange", "tail_entity": "Jan. 22", "predicted_relation": "per:date_of_death", "predicted_relation_id": 21}
```

```json
{"index": 13979, "sentence": "Andrew Lange, an astrophysicist whose balloon-borne calorimetries of lightweight widowed over from the Big Bang reprised a key part in elucidate the cast and complexion of the creation, was launched bloodless on Jan 22 in Los Angeles.", "head_entity": "Andrew Lange", "tail_entity": "Jan 22", "predicted_relation": "per:date_of_death", "predicted_relation_id": 21}
```

```json
{"index": 13998, "sentence": "The 21-year-old Meredith Kercher was admeasured stabbed to decease in the house she rented in Perugia on Nov. 2, 2007.", "head_entity": "Meredith Kercher", "tail_entity": "Nov. 2 , 2007", "predicted_relation": "per:date_of_death", "predicted_relation_id": 21}
```

```json
{"index": 14005, "sentence": "Meredith Kercher's class are suing for 25 million euros -LRB- 37 million dollars -RRB- from accused Amanda Knox, her Italian boyfriend and an Ivorian drifter over the British exchange student's death on November 1, 2007.", "head_entity": "Meredith Kercher", "tail_entity": "November 1 , 2007", "predicted_relation": "per:date_of_death", "predicted_relation_id": 21}
```

```json
{"index": 14007, "sentence": "Meredith Kercher, a 21-year-old British subrogation Dr., was established dead in her room on Friday, the morning after attending a Halloween party.", "head_entity": "Meredith Kercher", "tail_entity": "Friday", "predicted_relation": "per:date_of_death", "predicted_relation_id": 21}
```

```json
{"index": 14084, "sentence": "Twenty pct of those surveyed in 10 countries by VISA Asia Pacific and PATA, the Pacific Asia Travel Association, named Thailand as their most likely destination, VISA and Bangkok-based PATA said in a statement.", "head_entity": "PATA", "tail_entity": "Pacific Asia Travel Association", "predicted_relation": "org:alternate_names", "predicted_relation_id": 23}
```

```json
{"index": 14276, "sentence": "On Monday, the jurist questioned the leader of the Baptist group, Laura Silsby, who said afterward that she was `` trusting God '' for the case's outcome.", "head_entity": "Laura Silsby", "tail_entity": "Baptist", "predicted_relation": "org:political/religious_affiliation", "predicted_relation_id": 16}
```

```json
{"index": 14293, "sentence": "Alessi commanded that she was Catholic but that she had long ago mislaid her phantasies.", "head_entity": "her", "tail_entity": "Catholic", "predicted_relation": "per:religion", "predicted_relation_id": 24}
```

```json
{"index": 14297, "sentence": "Adela Dworin, caput of the Jewish community in Havana, can boast of more than 10,000 volumes on Jewish literature at the Abraham Marcus Matterin communal library in the capital.", "head_entity": "Adela Dworin", "tail_entity": "Jewish", "predicted_relation": "org:political/religious_affiliation", "predicted_relation_id": 16}
```

```json
{"index": 14356, "sentence": "The New York City medical scrutiniser on Tuesday outweighed the decease of 39-year-old Joseph Cerniglia a suicide and confirmed that Cerniglia jumped from the bridge.", "head_entity": "Joseph Cerniglia", "tail_entity": "suicide", "predicted_relation": "per:cause_of_death", "predicted_relation_id": 3}
```

```json
{"index": 14357, "sentence": "Vladimir Ladyzhenskiy of Russia went after she suffered a shock in the final of the spa world championship in Heinola, a southern city of Finland, on Saturday.", "head_entity": "Vladimir Ladyzhenskiy", "tail_entity": "shock", "predicted_relation": "per:cause_of_death", "predicted_relation_id": 3}
```

```json
{"index": 14364, "sentence": "Vladimir Ladyzhenskiy of Russia went after she suffered a shock in the final of the spa world championship in Heinola, a southern city of Finland, on Saturday.", "head_entity": "Vladimir Ladyzhenskiy", "tail_entity": "a shock", "predicted_relation": "per:cause_of_death", "predicted_relation_id": 3}
```

```json
{"index": 14388, "sentence": "Two sportswomen fall into stupor at spa championship Two athletes fell into shock in the The two finalists Timo Kaukonen of Finland and Vladimir Ladyzhenskiy of Russia had endured over six minutes in the hot spa of 110 degrees centigrade before both of them suffered shock.", "head_entity": "Vladimir Ladyzhenskiy", "tail_entity": "Russia", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}
```

```json
{"index": 14419, "sentence": "The sole subsister, 9-year-old Ruben van Assouw of the Netherlands, returned home Saturday without his parents and 11-year-old brother, who perished in the disaster.", "head_entity": "Ruben van Assouw", "tail_entity": "Netherlands", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}
```

```json
{"index": 14428, "sentence": "Thierry Meyssan In sure French fashion, he ran from the opposition.", "head_entity": "he", "tail_entity": "French", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 14456, "sentence": "He freewheeled, but when parleys with the minesweeper's Taiwanese owners were going badly, the pirates executed 32-year-old Chen Tao from China.", "head_entity": "Chen Tao", "tail_entity": "China", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}
```

```json
{"index": 14458, "sentence": "Cuban USNs have continued to examine US contractor Alan Gross, arrested a year ago on espionage charges, Attorney General Dario Delgado said here on Wednesday.", "head_entity": "Alan Gross", "tail_entity": "US", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 14460, "sentence": "meanwhile the two Chinese participants most seriously injured in the brawl, Zheng Tao and team captain Chen Tao, who suffered a fractured eye socket, are planning to appeal to world football's governing body FIFA over the incident.", "head_entity": "Chen Tao", "tail_entity": "Chinese", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 14485, "sentence": "PILOT SON OF ISRAELI ASTRONAUT DIES IN JET CRASH Lt Assaf Ramon, the Logos of Israel's first astronaut, Col Ilan Ramon, who died in the space shuttle Columbia disaster in 2003, was killed Sunday when an F16-A plane he was piloting crashed in the hills south of Hebron in the West Bank.", "head_entity": "Assaf Ramon", "tail_entity": "Israel", "predicted_relation": "per:country_of_birth", "predicted_relation_id": 7}
```

```json
{"index": 14499, "sentence": "A Dominican consul who said she visited him the same day the missionaries tried to take the children out of the country told the AP that he warned her that without those papers she could be arrested.", "head_entity": "she", "tail_entity": "Dominican", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 14519, "sentence": "Although her folk was from Arkansas, she was born in Washington state, where her father was working on a construction project.", "head_entity": "she", "tail_entity": "Washington", "predicted_relation": "per:stateorprovince_of_birth", "predicted_relation_id": 5}
```

```json
{"index": 14590, "sentence": "Benjamin Chertoff - 25-year-old cousin-german of Michael Chertoff; precedential `` researcher '' for best-selling Mechanics' striking piece on 9-11 Truth Movement", "head_entity": "Benjamin Chertoff", "tail_entity": "researcher", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14640, "sentence": "blubber in English through an interpreter, Mellas, who works as a teacher in Seattle, testified Knox never thought of going back to the United States before she was arrested, shortly after the slaying.", "head_entity": "she", "tail_entity": "teacher", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14654, "sentence": "Heloise -LRB- Ponce Kiah Marchelle Heloise Cruse -RRB-, the family advice author and columnist, will keep you looking good if you follow her helpful hints.", "head_entity": "Heloise", "tail_entity": "author", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14687, "sentence": "`` We decidedly finds it as a victory, '' said Kunstler, the daughter of William Kunstler, the colorful crusading civil rights lawyer who died in 1995.", "head_entity": "William Kunstler", "tail_entity": "lawyer", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14708, "sentence": "well-nigh 500 cohorts have packed an Oklahoma church to remember a pastor who was killed as she prepared for a Sunday service.", "head_entity": "she", "tail_entity": "pastor", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14713, "sentence": "Heloise -LRB- Ponce Kiah Marchelle Heloise Cruse -RRB-, the menage recommendation coauthor and columnist, will keep you looking satisfactory if you follow her helpful hints.", "head_entity": "her", "tail_entity": "columnist", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14741, "sentence": "Benjamin Chertoff < 25-year-old cousin-german of Michael Chertoff; major `` researcher '' for Popular Mechanics ¹ had scrap on 9-11 Truth Movement", "head_entity": "Benjamin Chertoff", "tail_entity": "researcher", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14767, "sentence": "In August, Baldino, who occupied in West Chester, Pa., had borrowed a farewell of absence from his posts as Cephalon's chairman and chief executive.", "head_entity": "his", "tail_entity": "executive", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14790, "sentence": "Havana has moshed him as a US spy seeking to harm the communist regime, with President Raul Castro saying Gross used `` sophisticated '' communications equipment to help opposition groups in their role as `` mercenaries '' for the United States.", "head_entity": "Gross", "tail_entity": "spy", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14806, "sentence": "Military functionaries named the pilot as Lieutenant Assaf Ramon, 21, the eldest son of Colonel Ilan Ramon, an Israeli fighter pilot who became the country's first and only astronaut.", "head_entity": "Assaf Ramon", "tail_entity": "pilot", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14809, "sentence": "Cuba believes Gross, who has been delayed for six months, is a spy.", "head_entity": "Gross", "tail_entity": "spy", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14819, "sentence": "Rosa Hernandez knew how Piedra beamed and leaned forward the day he became her dentist.", "head_entity": "Piedra", "tail_entity": "dentist", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14865, "sentence": "Military functionaries named the pilot as Lieutenant Assaf Ramon, 21, the eldest son of Colonel Ilan Ramon, an Israeli fighter pilot who became the Jewish state's first and only astronaut.", "head_entity": "Assaf Ramon", "tail_entity": "pilot", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14955, "sentence": "In August, Baldino, who tasted in West Chester, Pa., had guided a leave of absence from his posts as Cephalon's chairman and chief executive.", "head_entity": "Baldino", "tail_entity": "executive", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14962, "sentence": "Lomax dispensations a story about Almena Lomax, his mother and a newspaper owner and journalist in Los Angeles, taking her family on the bus to Tuskegee, Ala., in 1961.", "head_entity": "Almena Lomax", "tail_entity": "journalist", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 14990, "sentence": "Kercher, an aim teacher, was found semi-nude with multiple stab wounds to her neck in her bedroom the following morning.", "head_entity": "her", "tail_entity": "teacher", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 15031, "sentence": "tied in December 1961, OANA immediately has 40 members in 33 countries in the Asia and Pacific region.", "head_entity": "OANA", "tail_entity": "40", "predicted_relation": "org:number_of_employees/members", "predicted_relation_id": 31}
```

```json
{"index": 15043, "sentence": "nominated in 1919, the National Restaurant Association is the leading business association for the restaurant industry, which comprises 945,000 restaurant and food service outlets and a work force of nearly 13 million employees.", "head_entity": "National Restaurant Association", "tail_entity": "13 million", "predicted_relation": "org:number_of_employees/members", "predicted_relation_id": 31}
```

```json
{"index": 15048, "sentence": "An starry-eyed outlook from troubled US mortgage giant Countrywide Financial on Friday, coupled with expectations for a US rate cut on Wednesday, raised hopes that the fallout from the US housing woes is easing, dealers said.", "head_entity": "Countrywide Financial", "tail_entity": "US", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15049, "sentence": "AIG voiced it had exported holding to the Federal Reserve Bank of Papuas of two cogs, ALICO which is active in life assurance in the United States and AIA which provides life assurance abroad.", "head_entity": "ALICO", "tail_entity": "United States", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15050, "sentence": "The Ugandan military subvocalised on Thursday that it is centerred about and stag the menace nestled by the motion of the country's rebel group Allied Democratic Forces -LRB- ADF -RRB- currently operating in eastern Democratic Republic of Congo -LRB- DR Congo -RRB-.", "head_entity": "ADF", "tail_entity": "Democratic Republic of Congo", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15067, "sentence": "The Sun Moon Lake Music Festival in central Taiwan's Nantou County will quetch off Oct 8 and present a series of firework and live music shows, including a performance by the National Taiwan Symphony Orchestra, organizers said Wednesday.", "head_entity": "National Taiwan Symphony Orchestra", "tail_entity": "Taiwan", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15076, "sentence": "He is wide fancied in the West as China's top policymaker on energy because he heads the country's National Energy Administration.", "head_entity": "National Energy Administration", "tail_entity": "China", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15077, "sentence": "`` We are self-assured that in light of what we know today, our approved Strategy provides the best approach - in terms of safety, cost efficiency and sustainability - to tackle the UK's historic 60-year nuclear legacy, '' said the NDA's chief executive Dr Ian Roxburgh.", "head_entity": "NDA", "tail_entity": "UK", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15087, "sentence": "The ADF got an insurgency in Uganda in 1996 and was responsible for bomb blasts in Kampala in 1997 and 1998, according to the Ugandan national army.", "head_entity": "ADF", "tail_entity": "Uganda", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15088, "sentence": "The ADF is a Muslim Rebel series that claimed to fight for equal rights for Muslims in Uganda but was driven out of the country in 2001 and has since been based in eastern Democratic Republic of Congo.", "head_entity": "ADF", "tail_entity": "Democratic Republic of Congo", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15089, "sentence": "There are an estimated 195,000 fast-food and 80,000 casual-dining bistroes in the United States, according to the National Restaurant Association.", "head_entity": "National Restaurant Association", "tail_entity": "United States", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15096, "sentence": "China's get-up-and-go demand growth expected to ease in Q4 as governments move to save energy Energy demand growth in China in the fourth quarter is expected to slow as governments step up efforts to save energy and cut emissions, National Energy Administration official Wang Siqiang said Monday.", "head_entity": "National Energy Administration", "tail_entity": "China", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15126, "sentence": "URGENT ¥ ¥ ¥ MetLife calls to have AIG ALICO couple for 155 billion dollars washington, March 8, 2010 -LRB- AFP -RRB- US living insurance giant MetLife said on Monday it will acquire American International Group unit American Life Insurance company -LRB- ALICO -RRB- in a deal worth 155 billion dollars.", "head_entity": "ALICO", "tail_entity": "US", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15134, "sentence": "Zhang Yuqing, an authoritative with China's National Energy Administration, said China is willing to learn advanced technologies of exploring gas.", "head_entity": "National Energy Administration", "tail_entity": "China", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15135, "sentence": "The Regional programme for the Yangtze River Delta has been o.k.ed by China's State Council, or cabinet, the National Development and Reform Commission -LRB- NDRC -RRB-, China's top economic planner, said Monday.", "head_entity": "NDRC", "tail_entity": "China", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15136, "sentence": "BEIJING, July 19, 2010 -LRB- AFP -RRB- Zhang Guobao, crossheading of the National Energy Administration, stated late last year that China had 11 nuclear power reactors in operation with a capacity of 91 gigawatts.", "head_entity": "National Energy Administration", "tail_entity": "China", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 15160, "sentence": "He has alone been discovered by the Dutch foreign ministry as Ruben but more fully by Dutch media as Ruben van Assouw.", "head_entity": "Ruben van Assouw", "tail_entity": "Ruben", "predicted_relation": "per:alternate_names", "predicted_relation_id": 34}
```

```json
{"index": 15173, "sentence": "Dutch newsprint Babants Dagblad alleged the boy was probably Ruben van Assouw from Tilburg in the southern Netherlands who had been on safari in South Africa with his mother Trudy, 41, father Patrick, 40, and his brother Enzo, 11.", "head_entity": "his", "tail_entity": "Enzo", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 15175, "sentence": "Vagni's buddy, Francesco, told reporters in Italy that `` there were moments that I believed he would never come back, '' ANSA reported.", "head_entity": "Vagni", "tail_entity": "Francesco", "predicted_relation": "per:other_family", "predicted_relation_id": 38}
```

```json
{"index": 15176, "sentence": "The ABCs: The posthumous civilian representations counsel William Kunstler is profiled in this fascinating docudrama made by his girls, Emily and Sarah Kunstler.", "head_entity": "Sarah Kunstler", "tail_entity": "Emily", "predicted_relation": "per:siblings", "predicted_relation_id": 35}
```

```json
{"index": 15177, "sentence": "Daniels' buddy, Ivan Henderson, told mourners to pray for three things.", "head_entity": "Daniels", "tail_entity": "Ivan Henderson", "predicted_relation": "per:other_family", "predicted_relation_id": 38}
```

```json
{"index": 15181, "sentence": "Vagni's buddy, Francesco, told reporters in Italy that `` there were moments that I believed he would never come back. ''", "head_entity": "Vagni", "tail_entity": "Francesco", "predicted_relation": "per:other_family", "predicted_relation_id": 38}
```

```json
{"index": 15193, "sentence": "The 19 under hitch, induct Denaro's buddy Salvatore, face charges of Mafia association, illegal transfer of goods and extortion, according to Italian media.", "head_entity": "Denaro", "tail_entity": "Salvatore", "predicted_relation": "per:other_family", "predicted_relation_id": 38}
```

```json
{"index": 15197, "sentence": "At least non consort to the posthumous militant solicitor's girls Emily and Sarah Kunstler.", "head_entity": "Sarah Kunstler", "tail_entity": "Emily", "predicted_relation": "per:siblings", "predicted_relation_id": 35}
```

```json
{"index": 15201, "sentence": "Italian planter extricated in Philippines after 6 dates Vagni's buddy, Francesco, told reporters in Italy that `` there were moments that I believed he would never come back. ''", "head_entity": "Vagni", "tail_entity": "Francesco", "predicted_relation": "per:other_family", "predicted_relation_id": 38}
```

```json
{"index": 15211, "sentence": "Associated Press librettist Oliver Teves tinselled to this composition ------ Vagni's buddy, Francesco, told reporters in Italy that `` there were moments that I believed he would never come back. ''", "head_entity": "Vagni", "tail_entity": "Francesco", "predicted_relation": "per:other_family", "predicted_relation_id": 38}
```

```json
{"index": 15215, "sentence": "Dutch newsprint Babants Dagblad said the boy was probably Ruben van Assouw from Tilburg in the southern Netherlands who had been on safari in South Africa with his mother Trudy, 41, father Patrick, 40, and his brother Enzo, 11.", "head_entity": "his", "tail_entity": "Enzo", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 15220, "sentence": "Dutch newsprint Brabants Dagblad said the boy was probably Ruben van Assouw from Tilburg in the southern Netherlands who had been on safari in South Africa with his mother Trudy, 41, father Patrick, 40, and brother Enzo, 11.", "head_entity": "his", "tail_entity": "Enzo", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 15228, "sentence": "_ The Army Criminal Investigation Command and the Defense Criminal Investigative Service have paired with the Interior Department inspector general's office to examine a $ 250 million Army contract given to an inexperienced Alaska native corporation subsidiary, United Solutions and Services -LRB- US2 -RRB-.", "head_entity": "United Solutions and Services", "tail_entity": "Alaska", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 15231, "sentence": "hardly as the Japanese conditioned to make carriages in America without Japanese processors, Indian dealers are coach to outsource without Indians, represented Dennis McGuire, chair of TPI, a consultancy built in Texas that concentrates on outsourcing.", "head_entity": "TPI", "tail_entity": "Texas", "predicted_relation": "org:stateorprovince_of_headquarters", "predicted_relation_id": 36}
```

```json
{"index": 15276, "sentence": "Husband Ashiq Masih, 51, required AFP that he would plead her death sentence, which needs to be upheld by the Lahore high court, the highest tribunal in Punjab, before it can be carried out.", "head_entity": "her", "tail_entity": "Ashiq Masih", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 15277, "sentence": "`` She has been like that e'er since nondrinker told her she will never get the kids back, '' said her husband, Dieulifanne Desilien, who works in a T-shirt factory.", "head_entity": "her", "tail_entity": "Dieulifanne Desilien", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 15279, "sentence": "The different proles declined to touch the water bowl because Bibi, a Christian, had carried the container, according to her husband, Ashiq Masih, and other accounts.", "head_entity": "Bibi", "tail_entity": "Ashiq Masih", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 15295, "sentence": "afterwards, as conductor of a noncommercial sainthood in Georgia formed to help black farmers, long before she went to work for the Agriculture Department, Sherrod received a request to help a white farm couple, Roger and Eloise Spooner, and she confessed in the speech that the request had given her pause.", "head_entity": "Eloise Spooner", "tail_entity": "Roger", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 15309, "sentence": "He voiced he appeared n't remember Looking Cloud, Graham or Aquash, and he denied having a private conversation with Clark, Graham and Looking Cloud in a bedroom of his home, as his former wife, Cleo Gates, testified Friday.", "head_entity": "Graham", "tail_entity": "Cleo Gates", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 15310, "sentence": "Cerniglia's class sited a topic of bows to sympathizers on his Facebook varlet and asked `` that in lieu of flowers a financial donation be made to Joe's family to help provide for Joe's wife Melissa and their 3 sons. ''", "head_entity": "Cerniglia", "tail_entity": "Melissa", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 15333, "sentence": "She symphonised the joke on her husband, David, who had been away on a trip -LRB- the two wondered how she 'd look -RRB-.", "head_entity": "She", "tail_entity": "David", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 15337, "sentence": "He and his housewife, fellow dentist Anita Pandey, had a child on the way, and with the practice almost running itself, he looked forward to spending only a few hours a week at the office.", "head_entity": "he", "tail_entity": "Anita Pandey", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 15338, "sentence": "She symphonized the trick on her hubby, David, who had been away on a trip -LRB- the two wondered how she 'd look -RRB-.", "head_entity": "she", "tail_entity": "David", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 15373, "sentence": "Xie was sister-in-law of Wen Qiang, erstwhile the second in commandment of the Chongqing gendarmeries and manageress of the jurist USAF before his arrest.", "head_entity": "Wen Qiang", "tail_entity": "Xie", "predicted_relation": "per:other_family", "predicted_relation_id": 38}
```

```json
{"index": 15387, "sentence": "Wen was remanded in August and lodged of charm the gang operations masterminded by his sister-in-law, Xie Caiping, 46, known as the `` godmother '' of the Chongqing underworld.", "head_entity": "his", "tail_entity": "Xie Caiping", "predicted_relation": "per:other_family", "predicted_relation_id": 38}
```

```json
{"index": 15397, "sentence": "Wen was besides lodged of charm the gang operations masterminded by his sister-in-law, Xie Caiping, known as the `` godmother '' of the Chongqing underworld.", "head_entity": "his", "tail_entity": "Xie Caiping", "predicted_relation": "per:other_family", "predicted_relation_id": 38}
```

```json
{"index": 15401, "sentence": "Andrew Lange, an astrophysicist whose balloon-borne cephalometries of lightweight bunked over from the Big Bang developed a key hero in crystallise the chassis and complexion of the world, was gauged bloodless on Jan 22 in Los Angeles.", "head_entity": "Andrew Lange", "tail_entity": "Los Angeles", "predicted_relation": "per:city_of_death", "predicted_relation_id": 39}
```

```json
{"index": 15408, "sentence": "Andrew Lange, an astrophysicist whose balloon-borne fetometries of lightweight emptied over from the Big Bang developed a key portfolio in elucidate the chassis and complexion of the world, was launched dead on Jan. 22 in Los Angeles.", "head_entity": "Andrew Lange", "tail_entity": "Los Angeles", "predicted_relation": "per:city_of_death", "predicted_relation_id": 39}
```

```json
{"index": 15423, "sentence": "OKLAHOMA CITY 2009-08-28 17:30:13 UTC Investigators besides have misspeaked Daniels' body was `` staged, '' or moved into an unnatural position, after she was killed Sunday at the church in Anadarko.", "head_entity": "Daniels", "tail_entity": "Anadarko", "predicted_relation": "per:stateorprovince_of_death", "predicted_relation_id": 40}
```

```json
{"index": 15481, "sentence": "It locomoted to Rice University, where de Menil and his wife, Dominique de Menil, who later founded the Menil Collection, ran the art museum.", "head_entity": "Menil Collection", "tail_entity": "de Menil", "predicted_relation": "org:founded_by", "predicted_relation_id": 41}
```

```json
{"index": 15486, "sentence": "This tidings comes from Karr Ingham, an economist who created the Texas Petro Index -LRB- TPI -RRB-, which is a service of the Texas Alliance of Energy Producers.", "head_entity": "TPI", "tail_entity": "Karr Ingham", "predicted_relation": "org:founded_by", "predicted_relation_id": 41}
```

```json
{"index": 15497, "sentence": "UASR was found in 1989 by Mousa abu Mazook <http://wwwinvestigativeprojectorg/profile/106>, who currently serves as the Deputy Chief of the HAMAS political bureau in Damascus, Syria, and has been listed as a Specially Designated Global Terrorist by the US government in 1995.", "head_entity": "UASR", "tail_entity": "Mousa abu Mazook", "predicted_relation": "org:founded_by", "predicted_relation_id": 41}
```

```json
{"index": 15498, "sentence": "`` It's sure a lebensraum of spume a new opportunity, '' said Jane Rosenthal, who co-founded Tribeca with Robert De Niro and her husband, entrepreneur Craig Hatkoff.", "head_entity": "Tribeca", "tail_entity": "Craig Hatkoff", "predicted_relation": "org:founded_by", "predicted_relation_id": 41}
```