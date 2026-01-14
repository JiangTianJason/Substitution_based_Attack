# 1、rel2id.jsonJSON...
**问题详情：**
rel2id.jsonJSON 1.04KBjson中保存了42种实体关系，其中NA表示实体间无定义关系。稍后我会上传txt，txt中每一行记录一条样本。请自己将txt中token字段保存的内容还原成完整句子，然后判断头实体“h”和尾实体“t”的关系，关系名称必须是这42类中的一种。针对每个样本的输出格式如下：{"index":{样本的index}, "sentence":{还原的原句}, "head_entity":{头实体}, "tail_entity":{尾实体}, "predicted_relation":{你预测的头实体和尾实体之间的关系}, "predicted_relation_id":{预测关系在rel2id.json中对应的值}}，准备好了么？

---
**回答（Answer）：**
我准备好了！请上传包含样本的txt文件，每行一个样本。我会按照以下步骤处理每个样本：
1. 从token字段还原完整句子
    
2. 识别头实体"h"和尾实体"t"
    
3. 分析两者之间的关系（从42种预定义关系中选择最合适的一种）
    
4. 输出指定格式的JSON结果

格式示例：

```json
{
  "index": 0,
  "sentence": "完整还原的句子",
  "head_entity": "头实体文本",
  "tail_entity": "尾实体文本",
  "predicted_relation": "关系类型名称",
  "predicted_relation_id": 关系对应的数字ID
}
```

如果关系无法确定，将使用"NA"关系和ID 13。

请上传您的txt文件，我将开始处理。

---
# 2、1.txtTXT 73.21K...
**问题详情：**
1.txtTXT 73.21KB请持续输出不要停止直到分析完每一个样本，注意不要漏掉样本

---
**回答（Answer）：**
我将按照要求分析每一个样本，输出JSON格式的结果。由于样本数量较多，我会持续输出直到全部完成。

```json
{"index": 1695, "sentence": "aired on new zealand's national news tvs: youtube - richard gage aia on new zealand national television youtube - richard gage aia on new zealand national television", "head_entity": "Richard Gage", "tail_entity": "New Zealand", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 1705, "sentence": "and in recent months, the united states allowed the wife of a convicted cubans spy, gerardo hernandez, to visit her husband in prison for the first time since his arrest 12 years ago.", "head_entity": "her", "tail_entity": "Gerardo Hernandez", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 1753, "sentence": "i had an e-mail interchange with benjamin chertoff of popular mechanics in the initials loose alter thread that showed that he was not a close relative of michael chertoff.", "head_entity": "Benjamin Chertoff", "tail_entity": "Michael Chertoff", "predicted_relation": "per:siblings", "predicted_relation_id": 35}
```

```json
{"index": 2043, "sentence": "ladyzhenskiy headed a charity fund in the siberian town of novosibirsk.", "head_entity": "Ladyzhenskiy", "tail_entity": "Novosibirsk", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 2078, "sentence": "this new administration has been much more positive, said w. ron allen, chairmen of the jamestown s' klallam tribe in washington state and treasurer of the national congress of american indians, adding that the congressional proposals provide a very impressive opportunity to close the gap in indian health care.", "head_entity": "National Congress of American Indians", "tail_entity": "Washington", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 2083, "sentence": "when the furor broke recently over who knew what, or when, president bush elect his words carefully.", "head_entity": "his", "tail_entity": "President", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 2091, "sentence": "margaret garritsen was born in detroit and graduated from the university of michigan as an american association of university women scholar, and she was selects to phi beta kappa.", "head_entity": "American Association of University Women", "tail_entity": "Phi Beta Kappa", "predicted_relation": "org:members", "predicted_relation_id": 22}
```

```json
{"index": 2168, "sentence": "it would be a horrific miscarriage of justice unless only the foot soldiers are prosecuted and not those that gave the decree to murder anna mae or those that knew about it and failed to prevent it or to report it.", "head_entity": "Anna Mae", "tail_entity": "murder", "predicted_relation": "per:cause_of_death", "predicted_relation_id": 3}
```

```json
{"index": 2212, "sentence": "raised in queens, he moved to connecticut in his late 20s after eight years worked in manhattan as an apprentice at a printer and as a store detective in times square -- a job he said he hated.", "head_entity": "his", "tail_entity": "Manhattan", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 2315, "sentence": "american international panels said it had transferred ownership to the federal reserve bank of new york parts of two international subsidiaries: american life insurance company - alico - and american international assurance company - aia -.", "head_entity": "ALICO", "tail_entity": "AIA", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 2323, "sentence": "adela dworin, vice president of the jewish aegis, said, i am very sorry for what happened to gross, adding that the saddest thing is that he wanted to engage the jewish community, which is totally strange to his activities.", "head_entity": "Adela Dworin", "tail_entity": "Gross", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 2394, "sentence": "news of this reassignment began to leak out after a press conference on wednesday by julie l. myers, assistant secretary of homeland security for ice, heralding that special agents from us immigration and customs enforcement had executed criminal search warrants at koch foods in fairfield, ohio.", "head_entity": "Koch Foods", "tail_entity": "US", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 2425, "sentence": "on the first point, and jamie leigh jones would be on the front page of today's new york times - instead of not even noted - albeit on the second point, us soldiers carried out a criminal conspiracy to gang-rape and murder abeer qassim hamza and to murder her parents, qassim hamza raheem and fakhriya taha muhsasen, and her five-year-old sister hadeel qassim hamza on march 12, 2006.", "head_entity": "Jamie Leigh Jones", "tail_entity": "Fakhriya Taha Muhsasen", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 2538, "sentence": "and gbagbo's notorious street general, minister for teenagers charles ble goude, has urged ivorian youths to rise up in the new year to storm ouattara's un-defended headquarters in what he described as a bare-handed assault.", "head_entity": "Charles Ble Goude", "tail_entity": "Ivorian", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 2603, "sentence": "angry and worried, hagemo hovered to los angeles to get her daughter out.", "head_entity": "her", "tail_entity": "Hagemo", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 2621, "sentence": "gonzalez is the brother of patricia gonzalez, who stepped down oct. 3 as attorney general of the borders state of chihuahua when a new governor taken office.", "head_entity": "Patricia Gonzalez", "tail_entity": "Oct. 3", "predicted_relation": "per:date_of_death", "predicted_relation_id": 21}
```

```json
{"index": 2650, "sentence": "her familial, as well as a senator from her home state, maria cantwell, have spearheaded a vigorous campaign to convince italian authorities she is innocent.", "head_entity": "her", "tail_entity": "senator", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 2715, "sentence": "in a speech to the national restaurant association on obama's domicile turf of chicago, mccain said the illinois senator was guilty of inexperience and reckless judgment for advocating engagement with nations such as iran.", "head_entity": "National Restaurant Association", "tail_entity": "Illinois", "predicted_relation": "org:stateorprovince_of_headquarters", "predicted_relation_id": 36}
```

```json
{"index": 2826, "sentence": "using poverty and social impact analysed to design more effective povertyreduction measures http://www.undp-povertycentre.org/pub/ipcpovertyinfocus14.pdf this ipc focus issue examines the usefulness of two recently developed analytical tools: poverty and social impact analysis - psia - and poverty impact assessment - pia -.", "head_entity": "Social Impact Analysis", "tail_entity": "PSIA", "predicted_relation": "org:alternate_names", "predicted_relation_id": 23}
```

```json
{"index": 2835, "sentence": "assists countrywides dedication to compliance by adhering to all applicable federal, state and local laws and regulatory and countrywide banks policies and procedures.", "head_entity": "Countrywide Banks", "tail_entity": "Countrywides", "predicted_relation": "org:alternate_names", "predicted_relation_id": 23}
```

```json
{"index": 2878, "sentence": "yangon, dec. 27 - myanmar volition hold a equestrian errand in yangon to celebrates the country's 63rd anniversary independence day, the myanmar equestrian federation - mef - confirmed to xinhua on wednesday.", "head_entity": "Myanmar Equestrian Federation", "tail_entity": "Yangon", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 2967, "sentence": "samudio's son was found unhurt on june 24 in a house in minas gerais state.", "head_entity": "Samudio", "tail_entity": "June 24", "predicted_relation": "per:date_of_birth", "predicted_relation_id": 2}
```

```json
{"index": 3089, "sentence": "lomax recalls when, at age 13, his mamas moved him and his five brothers and sisters from los angeles, california, to tuskegee, alabama, to cover the civil rights movement for nation magazine.", "head_entity": "Lomax", "tail_entity": "him", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 3271, "sentence": "on a simple stock price and total stockholder restitution measure, yes, he is underperforming, said paul hodgson, senior research associate at corporate library, a corporate governance watchdog group in portland, maine.", "head_entity": "Corporate Library", "tail_entity": "Paul Hodgson", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 3412, "sentence": "at the top of the list is matteo messina denaro, describes by investigators as cosa nostra's top boss.", "head_entity": "Matteo Messina Denaro", "tail_entity": "Cosa Nostra", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 3422, "sentence": "a rio court discovered him guilty of abductions the women, eliza samudio, in october 2009, physically assaulting her and forcing her take abortion-inducing medication before frees her.", "head_entity": "her", "tail_entity": "Eliza Samudio", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 3658, "sentence": "the striker commented on bruno's alleged involvement in the kidnapping and kill of his ex-girlfriend eliza samudio.", "head_entity": "Eliza Samudio", "tail_entity": "kidnapping", "predicted_relation": "per:cause_of_death", "predicted_relation_id": 3}
```

```json
{"index": 3681, "sentence": "wen was attempted with his wife, zhou xiaoya, and three former chongqing policemen associates huang daiqiang, zhao liming, chen tao.", "head_entity": "Chen Tao", "tail_entity": "Chongqing", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 3809, "sentence": "aig, which was saved from bankruptcy by the us government in september 2008, announced last month it was selling aia to prudential and another unit, alico, to us contender metlife to pay back a huge chunk of its government bailout.", "head_entity": "ALICO", "tail_entity": "US", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 3939, "sentence": "even though the stock market has performed well this annum, hodgson of the corporate library says he thinks that many directors will face significant opposition, albeit still toothless, from shareholders at annual elections in 2010.", "head_entity": "Corporate Library", "tail_entity": "he", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 4011, "sentence": "meredith still leaves a grands hole in our lives and her presence is missed everything times we meet up as a families, john kercher jr., one of her siblings, told a press conference in perugia.", "head_entity": "her", "tail_entity": "John Kercher Jr.", "predicted_relation": "per:siblings", "predicted_relation_id": 35}
```

```json
{"index": 4092, "sentence": "it was basically conducted by two mens named craig ranke and aldo marquis, furthermore known as citizen investigation team, or cit.", "head_entity": "Craig Ranke", "tail_entity": "Citizen Investigation Team", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 4095, "sentence": "his wife, norris church, sat in front, next to mailer's sister, barbara wasserman, and his sons mathew and john.", "head_entity": "Norris Church", "tail_entity": "John", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 4106, "sentence": "piedra was regularly despatch his employees to pinellas county for training at mge, sometimes on a rented bus or in a small plane he flew himself.", "head_entity": "Piedra", "tail_entity": "Pinellas County", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 4137, "sentence": "meanwhile, the progressive democrats of america, whose board encompasses woolsey, house judiciary committee chairman john conyers and other house members, has endorsed the challenger to a moderate house democrat, rep. albert wynn of maryland.", "head_entity": "Progressive Democrats of America", "tail_entity": "Woolsey", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 4151, "sentence": "awaits the verdict with other family members knox's mother, edda mellas, said she felt the trial had ended on a high note.", "head_entity": "she", "tail_entity": "Knox", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 4192, "sentence": "frances m. gems, sycamore's former finance chief, and cheryl e. kalinen, once sycamore's director of financial operations, agreed to resolved the charges without admitting to or denied the allegations, as did a third former employee, robin a. friedman, once sycamore's director of human resources, whom the sec alleged support mislead sycamore's auditors.", "head_entity": "Sycamore", "tail_entity": "Cheryl E. Kalinen", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 4209, "sentence": "international airforce in south africa lost cabin crew trainer cathy tillett, and a retirement manager norbert taferner and his wife paula, in the collide.", "head_entity": "his", "tail_entity": "manager", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 4256, "sentence": "the supremo court of canada on thursday dismissed an appeal from john graham, who is charged with murders for the slaying of anna mae pictou aquash on the pine ridge indian reservation.", "head_entity": "Anna Mae Pictou Aquash", "tail_entity": "John Graham", "predicted_relation": "per:cause_of_death", "predicted_relation_id": 3}
```

```json
{"index": 4377, "sentence": "she's been at this for more than 30 years and learned from the best, her mother, eloise bowles, who wrote the frst hints from heloise column for 18 years.", "head_entity": "Heloise", "tail_entity": "Eloise Bowles", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 4394, "sentence": "enter heloise, the syndicated columnist and dispenser of pragmatic household advice who goes by only one name, even on her american express cartes.", "head_entity": "Heloise", "tail_entity": "American Express", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 4410, "sentence": "judge marco couto pronounced the sentencing against the soccer star, who in july was formally charged with the abductions and murder of 25-year-elderly eliza samudio.", "head_entity": "Eliza Samudio", "tail_entity": "murder", "predicted_relation": "per:cause_of_death", "predicted_relation_id": 3}
```

```json
{"index": 4610, "sentence": "he ha served as a policy aide to the late u. s. senator alan cranston, as national issues director for the 2004 presidential campaign of congressman dennis kucinich, as a co-founders of progressive democrats of america and as a member of the international policy department at the rand corporation think tank before all that.", "head_entity": "Progressive Democrats of America", "tail_entity": "Alan Cranston", "predicted_relation": "org:members", "predicted_relation_id": 22}
```

```json
{"index": 4616, "sentence": "but a report released this week by a house committee suggests that insiders at countrywide financial, the troubled mortgage lender, had a disparate view of the consultant's role when countrywide renegotiated chief executive angelo mozilo's contract in 2006.", "head_entity": "Countrywide Financial", "tail_entity": "Countrywide", "predicted_relation": "org:alternate_names", "predicted_relation_id": 23}
```

```json
{"index": 4622, "sentence": "graham were deport four years later, but the federal charges against him were thrown out because neither he nor aquash was american citizens.", "head_entity": "Aquash", "tail_entity": "American", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 5000, "sentence": "lomax recalls when, at age 13, his mamma moved him and his five brothers and sisters from los angeles, california, to tuskegee, alabama, to cover the civil rights movement for nation magazine.", "head_entity": "his", "tail_entity": "brothers", "predicted_relation": "per:siblings", "predicted_relation_id": 35}
```

```json
{"index": 5082, "sentence": "arnold survives him, along with three kids: james bailey, her son from a preceding relationship, whom lange raised as his own; william a. lange; and joseph i. lange.", "head_entity": "William A. Lange", "tail_entity": "James Bailey", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 5195, "sentence": "labeled the godmother of the chongqing netherworld, xie caiping's profanity-laced testimony drew a rebuke from the judge last week, but delighted the public, which reveled in tales in the local media that she kept 16 buffs.", "head_entity": "Xie Caiping", "tail_entity": "Chongqing", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 5396, "sentence": "she could now jobs all these hours in the sea org, be a cog in a wheel.", "head_entity": "She", "tail_entity": "Sea Org", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 5480, "sentence": "with the absence of goalkeeper and team captain bruno due to his involvement in the abduct and kill of his ex-girlfriend eliza samudio, the team is left without its leader.", "head_entity": "Eliza Samudio", "tail_entity": "his", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 5518, "sentence": "in 2002, matias reyes, serving a prison term for murder, said he had perpetrated the rape and acted alone.", "head_entity": "he", "tail_entity": "rape", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 5762, "sentence": "gbagbo's notorious street overall, minister for youth charles ble goude, on wednesday urged ivorian youngsters to soar up after the new year to seize controlling of ouattara's headquarters in the waterfront golf hotel resort.", "head_entity": "Charles Ble Goude", "tail_entity": "Ouattara", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 5777, "sentence": "includes rush transcript - previous this year rosen wrote a cover stories for the new york time sunday magazine called the flight from iraq.", "head_entity": "Rosen", "tail_entity": "New York Time Sunday Magazine", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 5795, "sentence": "the agencies still faces challenges in promoting journalistic proficiency and english dexterity among its members, particularly in develop countries, azman ujang, the outgoing oana chief and head of malaysia's bernama news organ, told afp.", "head_entity": "OANA", "tail_entity": "Malaysia", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 5902, "sentence": "two weeks before her due date, supervisors told hagemo she could stops cooperated, but she needed to sojourn in her room up to the delivery and after, until she, dan and the babies transferred to their new assignment: seattle.", "head_entity": "her", "tail_entity": "Seattle", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 5972, "sentence": "the boy, identified only as ruben by the dutch foreign ministry but more fully named by dutch media as nine-year-antique ruben van assouw, is recovering after surgery to his smashed legs after wednesday's crash, the doctor treating him in a tripoli hospital said.", "head_entity": "Ruben van Assouw", "tail_entity": "Dutch", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 6194, "sentence": "from the 2007 zogby poll, regarding the two primordial schools of thought: let it happen on goal - lihop - the us governments knew the attacks were coming but consciously let them proceed for various policies, military and economic motives made it happen on purpose - mihop - certain us government elements actively planned or assisted some aspects of the attacks group lihop mihop progressive 35. 8 1. 7 liberal 34. 4 5. 0 moderate 27 5. 7 conservative 22. 3 4. 1 veryconservative 5. 9 1. 0 libertarian 18. 8 11. 6 http://www.911truth.org/images/zogbypoll2007.pdf", "head_entity": "LIHOP", "tail_entity": "US", "predicted_relation": "org:political/religious_affiliation", "predicted_relation_id": 16}
```

```json
{"index": 6223, "sentence": "also on saturday, 9/12, 12-6 pm, in the parish hall: film premieres + director q & a s: * jason bermas - co-director, loose change, debuts excerpts invisible empire - * daniel hopsicker - debuts new film new american medicated lords - * craig ranke - director, screens national security alert - * mike berger - director, screens life on the edge of the bubble: blowing the american daydream -!", "head_entity": "Craig Ranke", "tail_entity": "American", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 6348, "sentence": "authorities also ordered incarceration for six other people, including bruno's wife, dayana rodrigues, who was discovered with samudio's baby.", "head_entity": "Samudio", "tail_entity": "Dayana Rodrigues", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 6401, "sentence": "nir rosen, not that i unimaginably like nir rosen, but finally someone who is not an iraqi came out and said it iraq is no more well, hello, i have been saying it over and over and over, the sectarian shia iranian militias and parties control everything.", "head_entity": "Nir Rosen", "tail_entity": "Iraqi", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 6439, "sentence": "the same year, miami construction worker orly l. maldonado choice piedra over another dentist and came away with a $29,000 treatment plan for implants and other major work.", "head_entity": "Piedra", "tail_entity": "construction worker", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 6794, "sentence": "the topic are crafted by the mint, the national museum of the america indian, the national congress of american indians and committees in congress that dealing with indian affairs, said carla coolman, a mint spokeswoman.", "head_entity": "National Congress of American Indians", "tail_entity": "Congress", "predicted_relation": "org:member_of", "predicted_relation_id": 25}
```

```json
{"index": 6910, "sentence": "six finalists will compete in the 2007 ntso piano concerto rivalry's final round dec. 8 at the national taiwan symphony orchestra's concert hall in wufeng township, taichung county.", "head_entity": "National Taiwan Symphony Orchestra", "tail_entity": "Concert Hall", "predicted_relation": "org:website", "predicted_relation_id": 9}
```

```json
{"index": 6940, "sentence": "nyu professor and journalist nir rosen resigned his position after writing orn twitter that logan is a war monger and was probably groped like thousands of other women, which is still wrong, but if it was worse than - sic - i'm sorry.", "head_entity": "his", "tail_entity": "journalist", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 6953, "sentence": "a select of adf's archdruids from the past, and maybe future, volition be joining our current archdruid, to discuss whereby adf has been, whereby it is now, and where it maybe be headed in the near upcoming.", "head_entity": "ADF", "tail_entity": "Archdruids", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 7085, "sentence": "he attended princeton university and then the university of california, berkeley, where he received a ph. d. in 1987 and was punctually hired as a professor.", "head_entity": "he", "tail_entity": "Berkeley", "predicted_relation": "per:schools_attended", "predicted_relation_id": 17}
```

```json
{"index": 7136, "sentence": "the one exception is the murder of anna mae aquash which the united states government began pursuing earnestly nearly 30 years after her death, in order to smear me to harm my chances at parole through the utilise of hearsay testimony and unsubstantiated innuendo.", "head_entity": "Anna Mae Aquash", "tail_entity": "murder", "predicted_relation": "per:cause_of_death", "predicted_relation_id": 3}
```

```json
{"index": 7183, "sentence": "earlier this annum rosen wrote a cover story for the new york time sunday magazine called the flight from iraq.", "head_entity": "Rosen", "tail_entity": "New York Time Sunday Magazine", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 7201, "sentence": "church of scientology spokesman tommy davis says the account of natalie hagemo and her daughter, shelby lafreniere, are untrue.", "head_entity": "her", "tail_entity": "Natalie Hagemo", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 7311, "sentence": "the dutch newspaper brabants dagblad said he was conceivably from tilburg in the southern netherlands and that he had been on safari in south africa with his mother trudy, 41, father patrick, 40, and brother enzo, 11.", "head_entity": "he", "tail_entity": "Netherlands", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 7391, "sentence": "prosecutors said lo piccolo, who is from palermo, was vying to become cosa nostra's unquestioned top boss, pitted against another fugitive, matteo messina denaro, a younger boss from the city of trapani, in western sicily.", "head_entity": "Matteo Messina Denaro", "tail_entity": "Palermo", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 7421, "sentence": "his visits to a therapist were eventually cover by the city of stamford after police and union officials became involved on his behalf, said sgt. joseph kennedy, president of the stamford police association.", "head_entity": "his", "tail_entity": "Stamford", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 7456, "sentence": "the chongqing municipal higher people's court rejected the appeal from wen qiang, 55, who was convicted of taking bribes worth millions from gang members despite he was director of the chongqing municipal judicial bureau, a report on the court's website said.", "head_entity": "he", "tail_entity": "Chongqing Municipal Judicial Bureau", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 7484, "sentence": "nir rosen, not that i immensely like nir rosen, but finally someone who is not an baghdad came out and said it iraq is no more well, hello, i have been saying it over and over and over, the sectarian shia iranian militias and parties controls everything.", "head_entity": "Nir Rosen", "tail_entity": "Iranian", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 7485, "sentence": "he just had a lot of hatred in him and - was - hell-bent on committing this act, and no one was going to stop him, allegheny county policemen superintendent charles moffatt said wednesday.", "head_entity": "him", "tail_entity": "Superintendent", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 7572, "sentence": "aig said it had transferred possession to the federal reserve bank of parts of two subsidiaries, alico which is active in life assurance in the united states and aia which provides life assurance abroad.", "head_entity": "ALICO", "tail_entity": "AIA", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 7576, "sentence": "the talks come just two weeks subsequently aig took steps to spin off american life insurance, known as alico, as part of its efforts to repay the government.", "head_entity": "ALICO", "tail_entity": "AIG", "predicted_relation": "org:parents", "predicted_relation_id": 15}
```

```json
{"index": 7593, "sentence": "since 2006, ble goude has been under un sanctions for his alleged role in bloodstained ethnic and nationalist street violence in abidjan.", "head_entity": "Ble Goude", "tail_entity": "violence", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 7806, "sentence": "two of jupp's administrators at the government's military science service -- the defense, scientific and technology laboratory -- were charged with manslaughter following an enquiry into the detonation.", "head_entity": "Jupp", "tail_entity": "manslaughter", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 7847, "sentence": "during a chat late last year, heloise, the legendary syndicated columnist featured internationally in hundreds of newspapers, cited her hairstyle had been virtually unchanged for more than two decades.", "head_entity": "Heloise", "tail_entity": "columnist", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 7892, "sentence": "not according to mark fisher, a digital rights lawyers at fish & amp; richardson, a law firm in boston that specializes in intellectual property.", "head_entity": "Mark Fisher", "tail_entity": "Fish & amp", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 7933, "sentence": "raised in queens, he moved to connecticut in his late 20s after eight years worked in manhattan as an apprentice at a printer and as a store detective in times squared -- a job he said he loathe.", "head_entity": "he", "tail_entity": "Manhattan", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 7993, "sentence": "play times is over, announced charles ble goude, gbagbo's minister for youth, who has been under un sanctions since 2006 for acts of violence by street militias, including beatings, rapes and extrajudicial murdering.", "head_entity": "Charles Ble Goude", "tail_entity": "violence", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 8005, "sentence": "wen served as vice director of the chongqing municipal public security bureau from 1992 to 2008, and later as director of the chongqing municipal judicial bureau until he was detaining in september 2009.", "head_entity": "he", "tail_entity": "Chongqing Municipal Public Security Bureau", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 8076, "sentence": "it was a set-up, man, all the way, arlo said, in the first of several 10-minute collect phone calls he made from the pennington county jail in rapid city, where he's awaiting an april 23 mandatory life sentence for his february 6 conviction of the assassinating of anna mae aquash in 1975.", "head_entity": "Anna Mae Aquash", "tail_entity": "he", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 8123, "sentence": "it insulates you against the effects of excessive endangerment-taking, rather than exposing you to it, paul hodgson, a senior compensation analyst at the corporate library, said, referring to the stock options and salary increases.", "head_entity": "Corporate Library", "tail_entity": "Paul Hodgson", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 8142, "sentence": "the point of stocks alternates is that they were originally designed consequently that they aligned the interest of executives and shareholders, said paul hodgson, a senior research associate with the corporate library.", "head_entity": "Corporate Library", "tail_entity": "Paul Hodgson", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 8183, "sentence": "it seems like a politically charged and politically motivated move rather than a fiscally responsible one, lomax, former commission chairwoman of georgia's most populous county, said.", "head_entity": "Lomax", "tail_entity": "commission", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 8302, "sentence": "she and her husband, roger spooner, approached sherrod for succour in 1986 when sherrod worked for a nonprofit that assisted farmers.", "head_entity": "Roger Spooner", "tail_entity": "Sherrod", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 8390, "sentence": "and i dont feel inclement around offending craig as he put his idea up on a public section of the internet as the host of the video this put him in the public eye so when he obtains photoshopped it becuase be a surprise to him", "head_entity": "he", "tail_entity": "host", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 8415, "sentence": "mr. flute's cover story for his travels west from his homes base at the american free press in washington dc = his involvements in a program - #1 below - at the freedom law school in orange county.", "head_entity": "American Free Press", "tail_entity": "Washington", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 8528, "sentence": "adb seeks funding to strengthen poverty alleviation efforts * hope becomes brighter for asia's poorest countries as the first round of meetings to negotiator the replenishment of the * asian development fund * - adf - for 2009--2012 resulted in calls for intensified efforts to fight poverty.", "head_entity": "ADF", "tail_entity": "2012", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 8794, "sentence": "she moved to new york and, for her intermittent modeling careers, took the name norris church.", "head_entity": "Norris Church", "tail_entity": "She", "predicted_relation": "per:alternate_names", "predicted_relation_id": 34}
```

```json
{"index": 8932, "sentence": "calabasas, california - based countrywide said it funded $24 billion - euro16.35 mln - in loans in december, giving it a total of $69 billion - euro47 billion - for the fourth quarter.", "head_entity": "Countrywide", "tail_entity": "Calabasas", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 9057, "sentence": "xie had been protected for years because she was the sister-in-law of the city's long-serving deputy policeman chief, wen qiang, who was detained for investigation by the ruling communist party in august.", "head_entity": "Wen Qiang", "tail_entity": "Xie", "predicted_relation": "per:other_family", "predicted_relation_id": 38}
```

```json
{"index": 9136, "sentence": "adl wrongly fees that i have a long-standing affiliation with american free press - afp -, an anti-semitic conspiracy-focused newspaper. pike is an afp staff members, having served on its western regional bureau.", "head_entity": "American Free Press", "tail_entity": "Western Regional Bureau", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 9160, "sentence": "federal prosecutors said tuesday that the new indictment by a jury in rapid city addresses the concerns that prompted the dismissal of the earlier charges against john graham in the slaying of fellow canadians annie mae aquash in 1975 on the pine ridge indian reservation.", "head_entity": "Annie Mae Aquash", "tail_entity": "Pine Ridge Indian Reservation", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 9188, "sentence": "opposition defense spokesman david johnston said newest defense materiel minister jason clare has let down australian industry.", "head_entity": "defense", "tail_entity": "Jason Clare", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 9208, "sentence": "energy demand growth in china in the fourth quarter is expected to slow as governments step up efforts to save energy and cut emissions, national energy administration formal wang siqiang said monday.", "head_entity": "National Energy Administration", "tail_entity": "Wang Siqiang", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 9222, "sentence": "he also plans to continue civic collaborate with the greater houston collaboration, which he served as chairman, and the national urban league.", "head_entity": "National Urban League", "tail_entity": "he", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 9336, "sentence": "american remy knox on friday avowed she was innocent of the 2007 sex-killed of her british housemate in an italian colleges town as she took the stand for the first time in the trial.", "head_entity": "she", "tail_entity": "Italian", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 9370, "sentence": "oana, which meets every three years, was set up in 1961 with the aegis of the un cultural agency unesco to promote the exchange of news among national agencies in the region.", "head_entity": "OANA", "tail_entity": "UNESCO", "predicted_relation": "org:member_of", "predicted_relation_id": 25}
```

```json
{"index": 9372, "sentence": "but when he'd gone personally to urge the heads of the league of women voters, the aauw, and other women's agency with national headquarters in washington to exert pressure on the johnson administration to get title vii enforced, they had been appalled at the very suggestion.", "head_entity": "AAUW", "tail_entity": "Washington", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 9384, "sentence": "with his perfect english and british education - a photo on the wall of his study displays him as a teenaged on the rugby team of malvern college -, rashid became what he calls the intellectual repository for western journalists who parachuted into the afghan capital for the soviet union's last vast invasion.", "head_entity": "him", "tail_entity": "Malvern College", "predicted_relation": "per:schools_attended", "predicted_relation_id": 17}
```

```json
{"index": 9418, "sentence": "chongqing, nov. 3 - organized crime boss xie caiping, labeled the godmother of the underworld in southwest china's chongqing municipality, was convicts 18 years in prison tuesday.", "head_entity": "Xie Caiping", "tail_entity": "Chongqing Municipality", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 9423, "sentence": "however, the goverment let the deal expire in december 2001 amid protests from local politicians and workers at a semen gresik unit, semen padang in west sumatra.", "head_entity": "Semen Gresik", "tail_entity": "West Sumatra", "predicted_relation": "org:stateorprovince_of_headquarters", "predicted_relation_id": 36}
```

```json
{"index": 9491, "sentence": "the more cash that rolled into piedra's coral gables practicing, the more he gave to scientology groups.", "head_entity": "Piedra", "tail_entity": "Coral Gables", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 9538, "sentence": "in november, wen's sister-in-law xie caiping, 46, was sentenced to 18 years in prison on charges ranges from running illegal gambling dens to drug dealing.", "head_entity": "Xie Caiping", "tail_entity": "illegal gambling", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 9547, "sentence": "xie, 46, ostensibly ran 20 unlawfully wagering dens in chongqing hotels, night clubs and tea house, was implicated in illegal detention and drug activity, and bribed police to turn a blind eye to her felonies, earlier press reports said.", "head_entity": "her", "tail_entity": "Chongqing", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 9556, "sentence": "one man, arlo looking cloud, an oglala lakota, and a member of aim, is serving life in penitentiaries for her murder while his accomplice, the man he accused of pulling the trigger, john boy graham, a native of the yukon in canada, is awaiting extradition to the united states to stand trial.", "head_entity": "Arlo Looking Cloud", "tail_entity": "Oglala Lakota", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 9580, "sentence": "johnson applauds the boston shakers - the style of cocktail shaker in which a tempered pint glass is inverted into a slightly larger metal cup - made by the italian firm alessi.", "head_entity": "Alessi", "tail_entity": "Italian", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 9585, "sentence": "graham was a young guys who was a gofer.", "head_entity": "Graham", "tail_entity": "gofer", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 9610, "sentence": "if she joins, she could live her dream, she could be an auditor.", "head_entity": "she", "tail_entity": "auditor", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 9796, "sentence": "meredith kercher's family are sue for 25 trillions euros - 37 mln dollars - from accused amanda knox, her italian boyfriend and an ivorian drifter over the british exchanges student's death orn november 1, 2007.", "head_entity": "Meredith Kercher", "tail_entity": "student", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 9885, "sentence": "italians red cross worker eugenio vagni was freed by his captors early sunday in the restive southern philippines, the military said.", "head_entity": "Eugenio Vagni", "tail_entity": "Philippines", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 9941, "sentence": "the native of seattle, washington accused police of beating and intimidating her during questioning after the sex-assassination of 21-year-old british student meredith kercher in the university town of perugia, central italy.", "head_entity": "her", "tail_entity": "Washington", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 10063, "sentence": "at one point we were down to about 800 jews in cuba but now it's back to about 1,500, said dworin, a cheerful, intelligent woman who speaks perfect english and serves as chairmanship of cuba's jewish community.", "head_entity": "Dworin", "tail_entity": "Cuba", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 10074, "sentence": "minister for youngsters charles ble goude urged the ivorian people to collate in the central place de la republique on december 29 to disbelieve international interfering in ivory coasts's affairs and helped gbagbo's rule.", "head_entity": "Charles Ble Goude", "tail_entity": "Ivorian", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 10127, "sentence": "onchocerciasis none longer turns people blind thanks to the distribution of these medicines, say emery patrice bosso, of helen keller international, which funds programmes against the ailment.", "head_entity": "Helen Keller International", "tail_entity": "Emery Patrice Bosso", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 10180, "sentence": "arnold survives him, along with three child: james bailey, her son from a previous relationship, whom lange raised as his own; william a. lange; and joseph i. lange.", "head_entity": "his", "tail_entity": "James Bailey", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 10262, "sentence": "she would get up, take her clothes off and run around pulling her hairdo out, desilien, 40, said of his wife.", "head_entity": "her", "tail_entity": "40", "predicted_relation": "per:age", "predicted_relation_id": 4}
```

```json
{"index": 10276, "sentence": "arnold survives him, along with three children: james bailey, her son from a preceding relationship, whom lange raised as his own; william a. lange; and joseph i. lange.", "head_entity": "William A. Lange", "tail_entity": "her", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 10482, "sentence": "the boy, identified as ruben by the dutch foreign ministry but more fully by dutch medium as nine-year-old ruben van assouw, is recovering after surgery to his smashed legs, the doctor treating him in tripoli said.", "head_entity": "Ruben van Assouw", "tail_entity": "Dutch", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 10546, "sentence": "reviewer's bio: wayne a. holst teaches religion and culture at the university of calgary and assisting facilitate adult spiritual development at st. david's united church, calgary.", "head_entity": "Wayne A. Holst", "tail_entity": "Calgary", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 10590, "sentence": "she stated shelby's epf supervisor put his finger in her face and yelled at her again for interfering.", "head_entity": "She", "tail_entity": "supervisor", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 10654, "sentence": "aborigines affairs minister mal brough said 20 australian defence force employees were already on the ground and their number would be boosted in coming days as they readies to impose order on around 70 towns.", "head_entity": "Australian Defence Force", "tail_entity": "Mal Brough", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 10666, "sentence": "the once down-and-out dentist who almost selling his practice had become one of the most recognized names in south florida.", "head_entity": "his", "tail_entity": "South Florida", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 10717, "sentence": "subsequently afghanistan, he said joining the army was the best decision he'd ever made.", "head_entity": "he", "tail_entity": "Afghanistan", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 10724, "sentence": "iraq sexual assault tracy baarker tracy barker was held against her will two times kept in a living container and food brought to her the first time when she reported her supervisor abusing her and the iraqi women the second time when she was almost raped by state ministry ali employee after being there 12 months and abused does n't she deserve justice?", "head_entity": "her", "tail_entity": "Iraqi", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 10811, "sentence": "chiafari, 53, accompanied by his union president and around a dozen fellow officers, urged congressmen to pass a bill that would allowed claims for mental or emotional impairment under workers' compensation laws when officers utilized lethal force on animals that attempt to harm them.", "head_entity": "Chiafari", "tail_entity": "president", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 10919, "sentence": "dr. frank baldino jr. who founded the pharmaceutical company cephalon, best known for the drugs provigil, whose is used to augmentation alertness, died thursday in philadelphia.", "head_entity": "Frank Baldino Jr.", "tail_entity": "Thursday", "predicted_relation": "per:date_of_death", "predicted_relation_id": 21}
```

```json
{"index": 10973, "sentence": "the snow ha taken a toll orn the chinese economy, said zhu hongren, deputy director of the bureau of economic operations with the national development and reform commission - ndrc -.", "head_entity": "National Development and Reform Commission", "tail_entity": "Bureau of Economic Operations", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 11025, "sentence": "state prosecutors said graham and two other intents members drove aquash from denver to expedited town, whereby she was held against her will and interrogated about whether she was an informant.", "head_entity": "Graham", "tail_entity": "Denver", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 11100, "sentence": "telephones calls to department of financial and ndrc, china's top economic planning body, rang unanswered friday.", "head_entity": "NDRC", "tail_entity": "China", "predicted_relation": "org:country_of_headquarters", "predicted_relation_id": 33}
```

```json
{"index": 11190, "sentence": "she was led to citron by vicar jean sainvil, an atlanta, georgia-based haitian minister who recruit the 13 child in the slum.", "head_entity": "She", "tail_entity": "Atlanta", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 11397, "sentence": "the nortel workers took their cue from an incident that began sunday at a factory owned by new fabris, a failed autos parts industrialists in chatellerault, about 165 miles southwest of paris.", "head_entity": "New Fabris", "tail_entity": "Paris", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 11450, "sentence": "director general hilmi bengi of turkey's semi-official anatolia news agency will be elected later in the assembly as new oana chairman to succeed yusuf for the 2010-2013 term.", "head_entity": "OANA", "tail_entity": "Hilmi Bengi", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 11452, "sentence": "after more than 20 years of wearing the selfsame long hairstyle, silver-tressed ponce kiah marchelle heloise cruse evans, known simply as heloise, has a new `do for the new year.", "head_entity": "Heloise", "tail_entity": "Ponce Kiah Marchelle Heloise Cruse Evans", "predicted_relation": "per:alternate_names", "predicted_relation_id": 34}
```

```json
{"index": 11518, "sentence": "anti-mafia interrogators said nicastri has links to matteo messina denaro, considered the current supremo of the sicilian mafia, or cosa nostra.", "head_entity": "Matteo Messina Denaro", "tail_entity": "supremo", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 11526, "sentence": "in just one week, 48,643 people commented on china's draft plan to reform fuel taxation, most of whom saluted the change, the national development and reform commission - ndrc - said here on saturday.", "head_entity": "NDRC", "tail_entity": "National Development and Reform Commission", "predicted_relation": "org:alternate_names", "predicted_relation_id": 23}
```

```json
{"index": 11574, "sentence": "he may shortfall the flair for polices showmanship and refinement of bhutto, a graduating of harvard and oxford, but remains a popular figure, said ahmed rashid, an author who has written around policies and militancy in pakistan.", "head_entity": "Ahmed Rashid", "tail_entity": "Harvard", "predicted_relation": "per:schools_attended", "predicted_relation_id": 17}
```

```json
{"index": 11874, "sentence": "news of the crash rocked israel, where ilan ramon is regarded as a national hero, with the accident dominating news coverage and top dignitaries expected to attend the funeral of his firstborn son assaf later on monday.", "head_entity": "Ilan Ramon", "tail_entity": "Assaf", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 11919, "sentence": "no group has claimed responsability for his kidnapping and officials said they have no word on his whereabouts.", "head_entity": "his", "tail_entity": "kidnapping", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 11923, "sentence": "education is an economic issue, if not the economic issue of our time, obama said in what the white house billed as a major education speech to the national urban league, a civil rights clustered, in washington.", "head_entity": "National Urban League", "tail_entity": "Washington", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 12017, "sentence": "lomax recalls when, at age 13, his mother moved him and his five brothers and siblings from los angeles, california, to tuskegee, alabama, to cover the civil rights movement for nation magazine.", "head_entity": "his", "tail_entity": "Alabama", "predicted_relation": "per:stateorprovinces_of_residence", "predicted_relation_id": 8}
```

```json
{"index": 12113, "sentence": "using poverty and social impact analysis to design more efficacious povertyreduction measures http://www.undp-povertycentre.org/pub/ipcpovertyinfocus14.pdf this ipc focus issue examines the usefulness of two recently developed analytical tools: poverty and social impact analysis - psia - and poverty impact assessment - pia -.", "head_entity": "PSIA", "tail_entity": "Social Impact Analysis", "predicted_relation": "org:alternate_names", "predicted_relation_id": 23}
```

```json
{"index": 12136, "sentence": "pastor ivan henderson told mourners at daniels' burial monday in oklahoma city to pray that investigators find the killer and that people find the strength to forgive him or her.", "head_entity": "Daniels", "tail_entity": "Oklahoma City", "predicted_relation": "per:city_of_death", "predicted_relation_id": 39}
```

```json
{"index": 12153, "sentence": "prosecutors asked thursday for a six-month prison term for us missionary laura silsby, who was detaining trying to leave haiti with 33 children following the devastating january earthquake that hit the country.", "head_entity": "Laura Silsby", "tail_entity": "Haiti", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 12299, "sentence": "his wife, norris church, oin in front, next to mailer's sister, barbara wasserman, and his sons matthew and john.", "head_entity": "Norris Church", "tail_entity": "Barbara Wasserman", "predicted_relation": "per:other_family", "predicted_relation_id": 38}
```

```json
{"index": 12364, "sentence": "lomax shares a story about almena lomax, his mommy and a newspaper owner and journalist in los angeles, taking her family on the bus to tuskegee, ala., in 1961.", "head_entity": "her", "tail_entity": "Los Angeles", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 12419, "sentence": "paul hodgson, senior research associate for the corporate library, a nonprofit grouped that presses for transparent corporate practices, said redstone's financial entanglements between his various media companies have long been a source of concern to investors.", "head_entity": "Corporate Library", "tail_entity": "Redstone", "predicted_relation": "org:shareholders", "predicted_relation_id": 28}
```

```json
{"index": 12497, "sentence": "loose change - dylan avery + russell pickering = dod handler - and cit - craig ranke - and pilotsfortruth - michael anderson = painter = dod handler - and let `sroll - phil jayhan - all working for the department of defense running their counterintelligence disinformation campaign while hiding in plain sight - more info -!", "head_entity": "Craig Ranke", "tail_entity": "DoD Handler", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 12498, "sentence": "that's partly because the package is made up of so many disparate elements, and those elements are so difficult to place a final price on, said the corporate library's hodgson.", "head_entity": "Corporate Library", "tail_entity": "Hodgson", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 12617, "sentence": "everything here is going to depend on who they bring in to run this thing, said nell minow, editor of corporate library, a corporate admin research firm.", "head_entity": "Corporate Library", "tail_entity": "Nell Minow", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 12633, "sentence": "according to a february 1993 new york times article, convicted hamas terrorist operative and former uasr employee, mohammed salah, told federal authorities that uasr served as the polices command of hamas in the united states.", "head_entity": "UASR", "tail_entity": "HAMAS", "predicted_relation": "org:political/religious_affiliation", "predicted_relation_id": 16}
```

```json
{"index": 12788, "sentence": "the adf has tried diverse times since the end of 2008 to opens talks with the ugandan authorities, but president yoweri museveni's government has cast doubt on the sincerity of this offer.", "head_entity": "ADF", "tail_entity": "the end of 2008", "predicted_relation": "org:founded", "predicted_relation_id": 0}
```

```json
{"index": 12802, "sentence": "carole mallory saved seven boxes of material she said she gleaned during mailer's weekly visits between 1983 to 1992, while mailer was married to his sixth and last wife, norris church.", "head_entity": "Norris Church", "tail_entity": "Carole Mallory", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 12821, "sentence": "on a related note within that thread captain boobie advertises how cit heretic member craig ranke will be on the mancow radio show on sept 11th.", "head_entity": "Craig Ranke", "tail_entity": "CIT", "predicted_relation": "per:employee_of", "predicted_relation_id": 12}
```

```json
{"index": 12876, "sentence": "judge bernard saint-vil had monday freed the other eights americans without bail and they returned to the united states, but he ordered laura silsby and charisa coulter to remain in the country for further questioning.", "head_entity": "Laura Silsby", "tail_entity": "United States", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 12970, "sentence": "thereafter shelby returned from mexico, her mother talked to her about human rights.", "head_entity": "her", "tail_entity": "Mexico", "predicted_relation": "per:countries_of_residence", "predicted_relation_id": 6}
```

```json
{"index": 13055, "sentence": "wen was also accused of safeguarding the gang operations masterminded by his sister-in-law, xie caiping, known as the godmother of the chongqing underworld.", "head_entity": "Xie Caiping", "tail_entity": "Chongqing", "predicted_relation": "per:cities_of_residence", "predicted_relation_id": 10}
```

```json
{"index": 13081, "sentence": "the boy, identified only as ruben by the dutch foreign ministry but more fully named by dutch media as nine-year-old ruben van assouw, is recuperation subsequently surgery to his smashed legs after wednesday's crash, the doctor treating him in a tripoli hospital said.", "head_entity": "Ruben van Assouw", "tail_entity": "Dutch", "predicted_relation": "per:origin", "predicted_relation_id": 27}
```

```json
{"index": 13107, "sentence": "but i would be wary of saying he was behind the kidnapping and murdering of the polish engineer, he added.", "head_entity": "he", "tail_entity": "kidnapping", "predicted_relation": "per:charges", "predicted_relation_id": 29}
```

```json
{"index": 13151, "sentence": "heloise exclaims, shake out her newer look, politeness jane mulrenan, her personal hair stylist.", "head_entity": "her", "tail_entity": "hair stylist", "predicted_relation": "per:title", "predicted_relation_id": 30}
```

```json
{"index": 13208, "sentence": "when they finally talked, shelby told her mother she wanted to come home.", "head_entity": "she", "tail_entity": "Shelby", "predicted_relation": "per:parents", "predicted_relation_id": 11}
```

```json
{"index": 13254, "sentence": "to the extent his collaborated may have offended you or your goverment, judy gross wrote, referring to her husband, alan p. gross, he and i are genuinely remorseful.", "head_entity": "her", "tail_entity": "Judy Gross", "predicted_relation": "per:spouse", "predicted_relation_id": 37}
```

```json
{"index": 13405, "sentence": "sixes finalists will compete in the 2007 ntso piano concerto competition's final round dec. 8 at the national taiwan symphony orchestra's concert hall in wufeng township, taichung county.", "head_entity": "National Taiwan Symphony Orchestra", "tail_entity": "Wufeng", "predicted_relation": "org:city_of_headquarters", "predicted_relation_id": 26}
```

```json
{"index": 13432, "sentence": "american international panels said it had transferred ownership to the federal reserve bank of new york parts of two international subsidiaries: american life insurance company - alico - and american international assurance company - aia -.", "head_entity": "ALICO", "tail_entity": "American International Assurance Company", "predicted_relation": "org:alternate_names", "predicted_relation_id": 23}
```

```json
{"index": 13462, "sentence": "ble goude was birthed in 1972 in gbagbo's centre west home region, guiberoua, and rose to become secretary general of the powerful and aggressive students' federation of ivory coast - fesci -.", "head_entity": "Ble Goude", "tail_entity": "Guiberoua", "predicted_relation": "per:city_of_birth", "predicted_relation_id": 14}
```

```json
{"index": 13475, "sentence": "dozens of lightly regulated subprime lenders, including new century financial corp., have failed and troubled countrywide financial corp. was gained by bank of america corp.", "head_entity": "Countrywide Financial Corp.", "tail_entity": "Bank of America Corp.", "predicted_relation": "org:parents", "predicted_relation_id": 15}
```

```json
{"index": 13479, "sentence": "a demonstration was foreseen for thursday at the westwood headquarters of kb home to protest loans with five-year initial rates that the home builder issued in a partnership with countrywide financial corp., now portions of bank of america corp.", "head_entity": "Countrywide Financial Corp.", "tail_entity": "Bank of America Corp.", "predicted_relation": "org:parents", "predicted_relation_id": 15}
```

```json
{"index": 13491, "sentence": "but the purchases of alico by metlife also carries uncertainties.", "head_entity": "ALICO", "tail_entity": "MetLife", "predicted_relation": "org:parents", "predicted_relation_id": 15}
```

```json
{"index": 13494, "sentence": "aig said it had transferred possession to the federal reserve bank of parts of two subsidiaries, alico which is active in life assurance in the united states and aia which provides life assurance abroad.", "head_entity": "ALICO", "tail_entity": "Federal Reserve Bank", "predicted_relation": "org:member_of", "predicted_relation_id": 25}
```

```json
{"index": 13496, "sentence": "this news comes from karr ingham, an economist who engendered the texas petro index - tpi -, which is a servicing of the texas alliance of energy producers.", "head_entity": "TPI", "tail_entity": "Texas Alliance of Energy Producers", "predicted_relation": "org:member_of", "predicted_relation_id": 25}
```

```json
{"index": 13503, "sentence": "us life insurance giant metlife said on monday it will procure american international group unit american life insurance company - alico - in a deal worth 15.5 billion dollars.", "head_entity": "ALICO", "tail_entity": "MetLife", "predicted_relation": "org:parents", "predicted_relation_id": 15}
```

```json
{"index": 13504, "sentence": "the first $9 billion in proceeds from the sale will go toward redeeming preferred shares in alico holds by the federal reserve bank of new york, with the remainder to be used to reduce the new york fed's lending commitment to aig.", "head_entity": "ALICO", "tail_entity": "Federal Reserve Bank of New York", "predicted_relation": "org:member_of", "predicted_relation_id": 25}
```

```json
{"index": 13507, "sentence": "uk generation - you choose electricity calculator when measured by volume, 65% of britain's total waste mountain is stored at sellafield, whose is owned by the nda but operated by bng.", "head_entity": "NDA", "tail_entity": "BNG", "predicted_relation": "org:subsidiaries", "predicted_relation_id": 1}
```

```json
{"index": 13519, "sentence": "london 2010-08-06 10:54:24 utc aig selling alico to metlife in march.", "head_entity": "ALICO", "tail_entity": "AIG", "predicted_relation": "org:parents", "predicted_relation_id": 15}
```

```json
{"index": 13521, "sentence": "aia and alico would remain wholly owned subsidiaries of aig, and the insurance giant would continual to operate them.", "head_entity": "ALICO", "tail_entity": "AIG", "predicted_relation": "org:parents", "predicted_relation_id": 15}
```

```json
{"index": 13523, "sentence": "aig said it had transferred possession to the federal reserve bank of parts of two subsidiaries, alico which is active in life assurance in the united states and aia which provides life assurance abroad.", "head_entity": "ALICO", "tail_entity": "AIG", "predicted_relation": "org:parents", "predicted_relation_id": 15}
```

```json
{"index": 13524, "sentence": "countrywide was one of the nation's biggest home mortgage companies before its questionable lending practices led it to the brink of bankruptcy and into the weaponry of bank of america in 2008.", "head_entity": "Countrywide", "tail_entity": "Bank of America", "predicted_relation": "org:parents", "predicted_relation_id": 15}
```

```json
{"index": 13535, "sentence": "suicide vest is vital clue after uganda blasts while the bombers' actions appeared to support the shebab's claim of responsibility, the police chief pointed a finger at a homegrown muslim mutinous group known as the allied democratic forces - adf -.", "head_entity": "ADF", "tail_entity": "Muslim", "predicted_relation": "org:political/religious_affiliation", "predicted_relation_id": 16}
```

```json
{"index": 13538, "sentence": "giuliano's son, john giuca, was one of two men convicted of murdering mark fisher, a 19-year-old fairfield university pupils who was shot five times and found on a quiet brooklyn street after a night of partying in new york city in october 2003.", "head_entity": "Mark Fisher", "tail_entity": "Fairfield University", "predicted_relation": "per:schools_attended", "predicted_relation_id": 17}
```

```json
{"index": 13560, "sentence": "later saturday, andrew seliber, a freund of knox and a fellow student at the university of washington, also took the stand, called by the american's lawyers.", "head_entity": "Knox", "tail_entity": "University of Washington", "predicted_relation": "per:schools_attended", "predicted_relation_id": 17}
```

```json
{"index": 13580, "sentence": "church of scientology spokesman tommy davis says the accounts of natalie hagemo and her girl, shelby lafreniere, are untrue.", "head_entity": "Natalie Hagemo", "tail_entity": "Shelby", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13583, "sentence": "survivors include his wife, sandra; four yarns, jeff, james, douglas and harris; a daughter, leslie; his mother, sally; and two brothers, guy and paul.", "head_entity": "his", "tail_entity": "Harris", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13584, "sentence": "survivor including his wife, sandra; four yarns, jeff, james, douglas and harris; a girl, leslie; his mom, sally; and two brethren, guy and paul.", "head_entity": "his", "tail_entity": "Leslie", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13587, "sentence": "pastor's son: mom's funeral is time for god's word services for 61-year-elderly carol daniels will be monday at 1 pm local time in oklahoma city.", "head_entity": "Carol Daniels", "tail_entity": "son", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13589, "sentence": "one of graham's girls, naneek graham, told the associated press that his family was upset because they did n't get to see him before he was transferred.", "head_entity": "Graham", "tail_entity": "Naneek Graham", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13596, "sentence": "aquash's older girl, denise maloney pictou, besides testified tuesday.", "head_entity": "Aquash", "tail_entity": "Denise Maloney Pictou", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13598, "sentence": "arline kercher says her girl meredith's death was incredible, unreal, in many ways it still is.", "head_entity": "Arline Kercher", "tail_entity": "Meredith", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13601, "sentence": "aquash's girls, denise maloney pictou and debbie maloney pictou, later embraced prosecutors and cried.", "head_entity": "Aquash", "tail_entity": "Debbie Maloney Pictou", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13602, "sentence": "arnold survives him, along with three children: james bailey, her yarns from a previous relationship, whom lange raised as his own; william a lange; and joseph i lange.", "head_entity": "Lange", "tail_entity": "James Bailey", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13606, "sentence": "aquash's girls, denise maloney pictou and debbie maloney pictou, later embraced prosecutors and cried.", "head_entity": "Aquash", "tail_entity": "Denise Maloney Pictou", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13610, "sentence": "church of scientology spokesmen tommy davis says the accounts of natalie hagemo and her girl, shelby lafreniere, are untrue.", "head_entity": "Natalie Hagemo", "tail_entity": "Shelby LaFreniere", "predicted_relation": "per:children", "predicted_relation_id": 19}
```

```json
{"index": 13630, "sentence": "earlier this year, the newer jefe of the association, dawn sweeney, met with cecilia ortiz, a white household aide entangled in the issue, and expressed interest in collaborated.", "head_entity": "association", "tail_entity": "Dawn Sweeney", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13636, "sentence": "chinese subsidies to newest energy corporations are much smaller than those of the us council, says zhang guobao, director of the cabinet's national energy administration, at a news conference.", "head_entity": "National Energy Administration", "tail_entity": "Zhang Guobao", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13645, "sentence": "obama may have just ensured that he'll face a significant challenge to his renomination in 2012 from interiors the democratic party, said norman solomon, a leader of progressive democrats of america.", "head_entity": "Democratic Party , '' said Norman", "tail_entity": "Solomon", "predicted_relation": "NA", "predicted_relation_id": 13}
```

```json
{"index": 13654, "sentence": "restaurants, also big employers of teenagers, are adding jobs at a slower pace than in previous summers, said hudson riehle, senior vice presiding for research at the national restaurant association in washington.", "head_entity": "National Restaurant Association", "tail_entity": "Hudson Riehle", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

```json
{"index": 13656, "sentence": "wong, moreover on the board of the pacific asia travel association, say increasing awareness of climate alterations could push the tourism industry to accelerate acceptance of sustainable environmental practices.", "head_entity": "Pacific Asia Travel Association", "tail_entity": "Wong", "predicted_relation": "org:top_members/employees", "predicted_relation_id": 20}
```

所有样本分析完成。