# 1、rel2id.jsonJSON...
**问题详情：**
rel2id.jsonJSON 1.75KBjson中保存了80种实体关系，稍后我会上传txt，txt中每一行记录一条样本。请自己将txt中token字段保存的内容还原成完整句子，然后判断头实体“h”和尾实体“t”的关系，关系名称必须是这80类中的一种。针对每个样本的输出格式如下：{"index":{样本的index}, "sentence":{还原的原句}, "head_entity":{头实体}, "tail_entity":{尾实体}, "predicted_relation":{你预测的头实体和尾实体之间的关系}, "predicted_relation_id":{预测关系在rel2id.json中对应的值}}，准备好了么？

---
**思考过程（Thinking）：**
首先，用户描述了一个任务：有一个JSON文件叫rel2id.json，它包含了80种实体关系，每个关系有一个ID。例如，"place served by transport hub": 0, 等等。

用户说稍后会上传一个txt文件，这个txt文件的每一行是一个样本。每个样本有一个"token"字段，我需要将这个token字段还原成完整的句子。然后，对于每个样本，我需要识别头实体"h"和尾实体"t"，并判断它们之间的关系。关系必须是rel2id.json中定义的80种之一。

输出格式是JSON对象，包含：
- "index": 样本的索引
    
- "sentence": 还原后的完整句子
    
- "head_entity": 头实体
    
- "tail_entity": 尾实体
    
- "predicted_relation": 预测的关系名称
    
- "predicted_relation_id": 预测关系在rel2id.json中对应的ID

用户问“准备好了么？”，意思是“准备好了吗？”，所以我应该确认我准备好了。

但在确认之前，我需要理解整个过程。用户还没有上传txt文件，所以我需要等待文件上传。一旦上传，我将处理每个样本。

步骤：
1. 用户上传txt文件。
    
2. 对于txt文件中的每一行（每个样本）：
    a. 解析样本：样本可能是一个JSON字符串或其他格式？用户说“txt中每一行记录一条样本”，并且有“token字段”。我需要查看样本的结构。
    b. 还原句子：token字段可能是一个列表或序列，我需要将它还原成完整的句子。例如，token字段可能是一个单词列表，我需要拼接成句子。
    c. 提取头实体和尾实体：用户提到“头实体'h'和尾实体't'”，所以样本中应该有这些字段。
    d. 预测关系：基于句子和实体，预测关系。关系必须是rel2id.json中的一种。
    e. 输出指定的JSON对象。

对于关系预测，用户没有提供训练模型或任何方法；我需要基于上下文或常识来判断关系？用户说“判断头实体和尾实体的关系”，但没有说明如何判断。可能样本中包含足够的信息，或者我需要使用一个预定义的模型？但在这个场景中，作为AI助手，我应该使用我的知识来预测关系。

用户说“请自己将txt中token字段保存的内容还原成完整句子，然后判断头实体'h'和尾实体't'的关系”，所以我有责任预测关系。既然没有模型，我必须基于实体和句子的上下文来推断关系。

但在输出中，我需要指定关系名称和ID，所以关系必须是rel2id.json中定义的。

现在，用户问“准备好了么？”，所以我应该回复确认我准备好了，并等待文件上传。

可能的回复：
- 确认我理解了任务。
    
- 表示我准备好了。
    
- 询问更多细节，如果需要。

但用户说“稍后我会上传txt”，所以我应该等待。

最终输出是针对每个样本的JSON对象。

在回复中，我可以说："是的，我准备好了。请上传txt文件，我将处理每个样本并输出结果。"

为了完整性，我应该确保我能处理文件。

由于这是一个文本交互，用户可能会粘贴文件内容或上传文件。在ChatGPT环境中，用户可能上传文件或提供内容。

在响应中，我可以说："I'm ready. Please upload the txt file or provide its content, and I will process each sample as described."

**回答（Answer）：**
是的，我完全准备好了！我已经理解了整个任务流程：
1. **rel2id.json 文件**：您已经提供了这个文件，其中包含 80 种预定义的实体关系及其对应的 ID（例如，`"place served by transport hub": 0`）。我将使用这个映射来确保所有预测的关系名称和 ID 都严格匹配这些定义。
    
2. **txt 文件处理**：
    - 稍后您上传的 txt 文件，每一行对应一个样本（样本应为 JSON 格式字符串，包含字段如 `"token"`、`"h"` 和 `"t"`）。
        
    - 对于每个样本，我会：
        a. **还原句子**：从 `"token"`         字段中提取内容（通常是一个 token 列表，如 `["token1", "token2", ...]`），并将其还原为完整的句子（通过拼接 token 并添加适当的空格和标点）。
        b. **提取实体**：从样本中直接提取头实体 `"h"`         和尾实体 `"t"`（这些字段应包含实体文本）。
        c. **预测关系**：基于还原后的句子、头实体和尾实体的上下文，推断它们之间的关系。关系名称必须来自 rel2id.json 的 80 种关系之一（例如，如果句子暗示头实体是尾实体的“country of citizenship”，则预测关系为 `"country of citizenship"`）。
        - **预测方法**：我将使用语义分析和上下文推理（基于常见知识库和实体类型），确保关系合理且一致。如果关系不明确，我会基于句子的主导线索进行最佳判断。
            d. **输出格式**：为每个样本生成一个 JSON 对象，包含 `index`（样本的索引，从 0 开始或基于样本中的索引字段）、`sentence`（还原后的句子）、`head_entity`（头实体文本）、`tail_entity`（尾实体文本）、`predicted_relation`（预测的关系名称）、`predicted_relation_id`（对应的 ID，从 rel2id.json 中查找）。
            
        
    
3. **输出示例**：
    假设一个样本的 `"token"`     为 `["John", "was", "born", "in", "France"]`，`"h"`     为 `"John"`，`"t"`     为 `"France"`，还原句子为 `"John was born in France."`，预测关系为 `"country of citizenship"`（ID 6）。输出将如下：
    ```json
    {
      "index": 0,
      "sentence": "John was born in France.",
      "head_entity": "John",
      "tail_entity": "France",
      "predicted_relation": "country of citizenship",
      "predicted_relation_id": 6
    }
    ```

### 下一步：
- **请上传 txt 文件**（您可以粘贴文件内容或直接上传文件）。文件内容应为每行一个 JSON 对象，例如：
    `{"token": ["token1", "token2", ...], "h": "head entity", "t": "tail entity", ...}`（可能包含其他字段如 `"index"`，但我会优先使用样本中的索引或按行号生成）。
    
- 一旦收到文件，我会立即逐行处理所有样本，并输出完整的 JSON 结果列表（每个样本一个 JSON 对象）。
    
- **如果有任何特殊要求或文件格式细节，请随时告知！** 我在这里等待您的上传。😊

---
# 2、0.txtTXT 13.95K...
**问题详情：**
0.txtTXT 13.95KB请持续输出不要停止直到分析完每一个样本，注意不要漏掉样本

---
**思考过程（Thinking）：**
我们有一个rel2id.json文件，其中定义了80种关系及其对应的id。
现在，我们有一个包含多个样本的文本文件（0.txt），每个样本是一个JSON对象，包含以下字段：
- index: 样本的索引
    
- token: 一个token列表，需要拼接成完整的句子（注意：可能需要处理空格和标点）
    
- h: 头实体
    
- t: 尾实体

任务：对于每个样本，我们需要：
1. 将token列表还原成完整的句子（sentence）。
    
2. 根据句子上下文，判断头实体（h）和尾实体（t）之间的关系（必须是rel2id.json中80种关系之一）。
    
3. 输出一个JSON对象，包含：
    - index: 样本的index
        
    - sentence: 还原后的句子
        
    - head_entity: 头实体（即h）
        
    - tail_entity: 尾实体（即t）
        
    - predicted_relation: 预测的关系名称（字符串，必须与rel2id中的关系名称完全一致）
        
    - predicted_relation_id: 该关系在rel2id.json中对应的id

注意：由于样本数量较多，我将逐个处理每个样本，并输出结果。

处理步骤：
a. 拼接token：将token列表中的元素用空格连接，但要注意一些标点符号前面不需要空格（如逗号、句号、引号等）。不过，由于原始token列表已经分好词，我们简单用空格连接通常即可，然后根据常见标点规则微调（例如，将 "," 改为 ","，但为了简单，我们先用空格连接，然后替换一些明显的问题）。
b. 关系预测：根据句子意思和头尾实体的类型，推断关系。这需要一定的常识和上下文理解。我会根据句子内容和实体类型选择最合适的关系。

由于关系有80种，我会参考rel2id.json中定义的关系。常见的如：
- 地理位置关系：如"located in the administrative territorial entity"（位于行政实体）
    
- 部分整体关系：如"part of"（部分）
    
- 创建关系：如"composer"（作曲家）等。

我们将逐一处理每个样本。

注意：如果关系不确定，我们需要根据上下文做最佳猜测。

开始处理：

**回答（Answer）：**


```json
{"index": 102, "sentence": "its provenance is on the slopes of mont aigoual in the cévennes mountains.", "head_entity": "mont aigoual", "tail_entity": "cévennes", "predicted_relation": "mountain range", "predicted_relation_id": 1}
{"index": 104, "sentence": "for the timorese conclusion much karwendel crests can be reached, for example the birkkarspitze from the south.", "head_entity": "birkkarspitze", "tail_entity": "karwendel", "predicted_relation": "mountain range", "predicted_relation_id": 1}
{"index": 313, "sentence": "stefaan minivan hecke is a belgian ( flemish ) politician.", "head_entity": "belgian", "tail_entity": "flemish", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
{"index": 403, "sentence": "this paragraphs mentions the advent nor corroborate processing throughout all productive nor ineffective cupboard nominees of hassan rouhani first and second administrations.", "head_entity": "first", "tail_entity": "hassan rouhani", "predicted_relation": "position held", "predicted_relation_id": 11}
{"index": 538, "sentence": "she did the old entail anchor for the new england sports network ' s sportsdesk news program and most recently the anchor on mlb network.", "head_entity": "sportsdesk", "tail_entity": "new england sports network", "predicted_relation": "part of", "predicted_relation_id": 67}
{"index": 644, "sentence": "he meanwhile released billy ray sirius ' s scrapbook \" wanna be your joe \" and re - released john cougar mellencamp ' s \" american fool \" and tom petty ' s \" greatest hits \".", "head_entity": "american fool", "tail_entity": "john cougar mellencamp", "predicted_relation": "performer", "predicted_relation_id": 9}
{"index": 660, "sentence": "the coverage resembles section 25 ' s \" from the hip \".", "head_entity": "from the hip", "tail_entity": "section 25", "predicted_relation": "performer", "predicted_relation_id": 9}
{"index": 697, "sentence": "it aided to the cartographer scrapbooks \" terror and magnificence \" ( 1996 ) by elvis costello and sarah leonard , and \" standing stone \" ( 1997 ) by paul mccartney.", "head_entity": "\"standing stone\"", "tail_entity": "paul mccartney", "predicted_relation": "composer", "predicted_relation_id": 51}
{"index": 878, "sentence": "the harbour of hampton roads is an important highway of commerce , especially for the cities of norfolk , portsmouth , and newport news.", "head_entity": "hampton roads", "tail_entity": "newport news", "predicted_relation": "contains administrative territorial entity", "predicted_relation_id": 4}
{"index": 925, "sentence": "the date did scarred per kaushik chakraborty ( percussionist from bangla gaza , prithibi ) through a acknowledgment concerto called ' hawae melechi pakhna ' at golf green , kolkata.", "head_entity": "prithibi", "tail_entity": "kolkata", "predicted_relation": "location", "predicted_relation_id": 58}
{"index": 934, "sentence": "at 1968 , bodast recorded an album for mgm records , opened for the who , and were the backing band for chuck berry at the royal albert hall in london.", "head_entity": "bodast", "tail_entity": "london", "predicted_relation": "location", "predicted_relation_id": 58}
{"index": 940, "sentence": "toward 1888 , otis concluded an hoist assay pagoda through their mill in yonkers , new york ; this was possibly the first elevator test tower in the united states.", "head_entity": "otis", "tail_entity": "yonkers, new york", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
{"index": 961, "sentence": "toward the concerto video the signal that pursey , the layover of sham 69 and a young boy are sitting next to is the sign for hersham road.", "head_entity": "sham 69", "tail_entity": "hersham", "predicted_relation": "location of formation", "predicted_relation_id": 13}
{"index": 995, "sentence": "xylokastro – evrostina ( ) represents a municipality in the corinthia regional unit , peloponnese , greece.", "head_entity": "xylokastro", "tail_entity": "peloponnese", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
{"index": 997, "sentence": "servants through the ambuklao nor binga dykes had to release waters after the storm , which flooded rice fields and fish ponds along the agno river.", "head_entity": "ambuklao", "tail_entity": "agno river", "predicted_relation": "tributary", "predicted_relation_id": 48}
{"index": 1120, "sentence": "all year the crypt plays host to the liverpool beer festival which attracts visitors , not only from all over uk but also europe and places such as the united states and australia.", "head_entity": "crypt", "tail_entity": "united states", "predicted_relation": "location", "predicted_relation_id": 58}
{"index": 1209, "sentence": "onto 2009 , jowood entertainment released the official website for \" arcania : gothic 4 \" , revealing new information , screenshots and artworks.", "head_entity": "arcania: gothic 4", "tail_entity": "jowood entertainment", "predicted_relation": "developer", "predicted_relation_id": 20}
{"index": 1254, "sentence": "he did a preliminary liberate date of april 2003 for japan , and would be developed by the same employees who worked on \" ace combat 2 \" at namco.", "head_entity": "ace combat 2", "tail_entity": "namco", "predicted_relation": "developer", "predicted_relation_id": 20}
{"index": 1256, "sentence": "the \" poxnora \" prenup aided premium jeux acquires at gare fund ( employed in all games published by sony online entertainment ).", "head_entity": "poxnora", "tail_entity": "sony online entertainment", "predicted_relation": "publisher", "predicted_relation_id": 17}
{"index": 1319, "sentence": "strongest from the officer congressmen whom served during \" pulseman \" must further work during the \" pikachu \" suite , embracing headmaster ken sugimori , designer satoshi tajiri , and composer junichi masuda.", "head_entity": "pulseman", "tail_entity": "ken sugimori", "predicted_relation": "architect", "predicted_relation_id": 15}
{"index": 1340, "sentence": "the yarn of wholesaler peter martin anker , he was the brother of nils anker and herman anker.", "head_entity": "nils anker", "tail_entity": "peter martin anker", "predicted_relation": "sibling", "predicted_relation_id": 33}
{"index": 1354, "sentence": "it got a grandson of landowner and politician peter martin anker , and nephew of landowner and politician nils anker and school founder herman anker.", "head_entity": "herman anker", "tail_entity": "peter martin anker", "predicted_relation": "sibling", "predicted_relation_id": 33}
{"index": 1361, "sentence": "it meanwhile trained the championship girl of affirmed , flawlessly.", "head_entity": "flawlessly", "tail_entity": "affirmed", "predicted_relation": "child", "predicted_relation_id": 78}
{"index": 1362, "sentence": "profiting for this situation , trần thủ độ arranged a spousal throughout chiêu hoàng nor trần cảnh , the eight - sunni - immemorial filho of trần thừa and the younger brother of trần liễu.", "head_entity": "trần liễu", "tail_entity": "trần thừa", "predicted_relation": "child", "predicted_relation_id": 78}
{"index": 1366, "sentence": "it makes married to anita bose pfaff ( born 1942 ) , meanwhile an economist faculty , nor the fille of indian nationalist subhas chandra bose.", "head_entity": "anita bose pfaff", "tail_entity": "subhas chandra bose", "predicted_relation": "child", "predicted_relation_id": 78}
{"index": 1373, "sentence": "fag moderator did a daugther from the broodmare boudoir , whose different heirs included daiwa serious , flower jar nor caracolero ' s lawd graustark.", "head_entity": "caracolero", "tail_entity": "graustark", "predicted_relation": "child", "predicted_relation_id": 78}
{"index": 1393, "sentence": "it makes the filho of phyllis ( culbertson ) and william henry draper iii and the younger brother of actress polly draper.", "head_entity": "polly draper", "tail_entity": "william henry draper iii", "predicted_relation": "child", "predicted_relation_id": 78}
{"index": 1394, "sentence": "still gerontius decided to betray constantine , and in 409 he proclaimed maximus as emperor in hispania.", "head_entity": "maximus", "tail_entity": "gerontius", "predicted_relation": "successful candidate", "predicted_relation_id": 25}
{"index": 1397, "sentence": "she makes the fille from emmy stipend - finalist editor , planter , nor compose , buz kohan , nor mama of twins jono and david , and jenji kohan.", "head_entity": "david", "tail_entity": "buz kohan", "predicted_relation": "child", "predicted_relation_id": 78}
{"index": 1569, "sentence": "wines on long island benefit from the moderating influence of the atlantic ocean , long island sound , and peconic bay on the local climate.", "head_entity": "long island sound", "tail_entity": "atlantic ocean", "predicted_relation": "part of", "predicted_relation_id": 67}
{"index": 1571, "sentence": "a mediterranean ambiance among believable fresh storms create optimal situations throughout sail , both off the coast in the indian ocean and in the swan river.", "head_entity": "swan river", "tail_entity": "indian ocean", "predicted_relation": "located in or next to body of water", "predicted_relation_id": 71}
{"index": 1580, "sentence": "lowa ( ) makes a u. s. sate in the midwestern united states , bordered by the mississippi river to the east and the missouri and big sioux rivers to the west.", "head_entity": "big sioux", "tail_entity": "missouri", "predicted_relation": "tributary", "predicted_relation_id": 48}
{"index": 1591, "sentence": "texas makes installed near the confluence of the bulkley river and morice river approximately south of smithers along highway 16.", "head_entity": "morice river", "tail_entity": "bulkley river", "predicted_relation": "tributary", "predicted_relation_id": 48}
{"index": 1594, "sentence": "the pes currents through the center of the district and into vologda oblast , where it joins the chagodoshcha , another major tributary of the mologa.", "head_entity": "pes", "tail_entity": "chagodoshcha", "predicted_relation": "tributary", "predicted_relation_id": 48}
{"index": 1605, "sentence": "this cantilever bridging system was first realized in 1867 at a bridge over the regnitz at bamberg and in the main bridge in haßfurt.", "head_entity": "regnitz", "tail_entity": "main", "predicted_relation": "crosses", "predicted_relation_id": 64}
{"index": 1847, "sentence": "toward 1999 pei integrated with marconi electronic systems , the defence interests of gec , at the same time changing its name from british aerospace to bae systems.", "head_entity": "marconi electronic systems", "tail_entity": "bae systems", "predicted_relation": "subsidiary", "predicted_relation_id": 39}
{"index": 1869, "sentence": "through every end is a white marble pavilion , called the sawan and bhadon pavilions , named after the hindu months , sawan and bhadon.", "head_entity": "sawan", "tail_entity": "bhadon", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
{"index": 1928, "sentence": "bizzarrini served upon 1964 throughout iso rivolta nor worded three mannequins : iso rivolta gt , iso grifo both a3l and a3c versions.", "head_entity": "iso grifo", "tail_entity": "iso rivolta", "predicted_relation": "subsidiary", "predicted_relation_id": 39}
{"index": 2034, "sentence": "towelhead makes a new written by alicia erian and first published april 6 , 2005.", "head_entity": "towelhead", "tail_entity": "alicia erian", "predicted_relation": "author", "predicted_relation_id": 17}
{"index": 2040, "sentence": "alois nebel makes a 2011 czechoslovak animated tragic theatre instructed per tomáš luňák , reasoned during the farcical - workbook trilogy by jaroslav rudiš and jaromír 99.", "head_entity": "alois nebel", "tail_entity": "jaroslav rudiš", "predicted_relation": "after a work by", "predicted_relation_id": 29}
{"index": 2051, "sentence": "the feather sexes makes a 1991 roman by the british adventurer sir ranulph fiennes ( reissued as killer elite , echoing the title of its 2011 film adaptation ).", "head_entity": "its 2011 film adaptation", "tail_entity": "sir ranulph fiennes", "predicted_relation": "after a work by", "predicted_relation_id": 29}
{"index": 2055, "sentence": "auteur james fenimore cooper , author of \" last of the mohicans \" , also wrote \" water witch \" , a novel which was inspired by his visits to highlands.", "head_entity": "last of the mohicans", "tail_entity": "james fenimore cooper", "predicted_relation": "author", "predicted_relation_id": 17}
{"index": 2057, "sentence": "he makes casually reasoned during the newer \" the night watch \" by sergei lukyanenko , and is the first part of a duology , followed by \" day watch \".", "head_entity": "day watch", "tail_entity": "sergei lukyanenko", "predicted_relation": "author", "predicted_relation_id": 17}
{"index": 2083, "sentence": "than the sidebar bureau chess from welsh ' s 2003 alive measure kino of dr seuss ' s \" the cat in the hat \" universal got cold feet and the project was again put into turnaround.", "head_entity": "the cat in the hat", "tail_entity": "seuss", "predicted_relation": "after a work by", "predicted_relation_id": 29}
{"index": 2093, "sentence": "the bourne caveat makes the third jas bourn newer written by robert ludlum and a sequel to \" the bourne supremacy \" ( 1986 ).", "head_entity": "the bourne supremacy", "tail_entity": "robert ludlum", "predicted_relation": "author", "predicted_relation_id": 17}
{"index": 2218, "sentence": "wwca operated accordingly outside from the immemorial hotel gary.", "head_entity": "wwca", "tail_entity": "gary", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
{"index": 2222, "sentence": "toward julio 2010 the counsellors from northbrook college submitted among the federal notifications commitee to assign wgaj ' s licensing to wfcr foundation , inc. , owner of wfcr in amherst , massachusetts.", "head_entity": "wfcr", "tail_entity": "amherst, massachusetts", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
{"index": 2229, "sentence": "wdoh makes the athletes diffusion arm for delphos jefferson and delphos saint johns sports.", "head_entity": "wdoh", "tail_entity": "delphos", "predicted_relation": "licensed to broadcast to", "predicted_relation_id": 31}
{"index": 2458, "sentence": "it already hosts the \" afternoon show \" on wdr 4 in germany.", "head_entity": "wdr 4", "tail_entity": "germany", "predicted_relation": "licensed to broadcast to", "predicted_relation_id": 31}
{"index": 2488, "sentence": "during another reception for cuban ambassador lianys torres rivera , president quang stressed that the state and people of vietnam treasured solidarity and comprehensive co - operation with cuba.", "head_entity": "president", "tail_entity": "vietnam", "predicted_relation": "head of government", "predicted_relation_id": 5}
```

---
# 3、1.txtTXT 14.39K...
**问题详情：**
1.txtTXT 14.39KB请持续输出不要停止直到分析完每一个样本，注意不要漏掉样本

---
**思考过程（Thinking）：**
我们继续处理第二个文件1.txt中的样本。同样，对于每一行，我们需要：
1. 解析JSON，提取index, token, h, t。
    
2. 将token列表还原成句子（注意：token列表中的字符串需要拼接，并注意空格和标点）。
    
3. 根据句子内容和头实体h、尾实体t的上下文，判断两者之间的关系（从80种关系中选择一种）。
    
4. 输出指定格式的JSON对象。

注意：关系名称和id必须严格匹配rel2id.json中的定义。

开始处理：

**回答（Answer）：**


```json
{"index": 2498, "sentence": "under his 2009 national day rally speech prime minister lee hsien loong characterized racial and religious divides as \" the most visceral and dangerous fault line \" in singapore society ,.", "head_entity": "prime minister", "tail_entity": "singapore", "predicted_relation": "head of government", "predicted_relation_id": 5}
{"index": 2600, "sentence": "toward rapid hedge of the gdańsk strike by western press , ( which was permitted into the shipyard ) , anna walentynowicz is mentioned earlier than lech wałęsa.", "head_entity": "anna walentynowicz", "tail_entity": "gdańsk", "predicted_relation": "work location", "predicted_relation_id": 38}
{"index": 2619, "sentence": "liberian chair ellen johnson sirleaf visited nairobi in december 2015 for a state visit and to attend the 10th ministerial conference of the wto.", "head_entity": "ellen johnson sirleaf", "tail_entity": "nairobi", "predicted_relation": "location", "predicted_relation_id": 58}
{"index": 2625, "sentence": "through the riverbank near elsinore , he is among the group of resistance fighters trying to cover the escape to sweden of an important scientist ( evidently the nuclear physicist niels bohr ).", "head_entity": "niels bohr", "tail_entity": "sweden", "predicted_relation": "country of citizenship", "predicted_relation_id": 6}
{"index": 2632, "sentence": "maurice already divides her period between brazil and the uk to teach at the roger gracie academy in london.", "head_entity": "roger gracie", "tail_entity": "london", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
{"index": 2640, "sentence": "the concentrator of the mission is the parish of the assumption of the blessed virgin mary which collaborates with the polish \" adam mickiewicz \" school in paris , next to the polish embassy.", "head_entity": "adam mickiewicz", "tail_entity": "paris", "predicted_relation": "location", "predicted_relation_id": 58}
{"index": 2650, "sentence": "ellsworth storey did among chicago , first visited seattle as a teenager , and settled there in 1903 after earning an architecture degree from the university of illinois.", "head_entity": "ellsworth storey", "tail_entity": "seattle", "predicted_relation": "residence", "predicted_relation_id": 37}
{"index": 2665, "sentence": "boost ' s pater , walter foster ( born fast ) , was born in vienna in 1923 and arrived in london as a refugee following kristallnacht.", "head_entity": "walter foster", "tail_entity": "london", "predicted_relation": "residence", "predicted_relation_id": 37}
{"index": 2763, "sentence": "the 3rd recent jersey volunteer infantry was an american civil war union army regiment of infantry from new jersey that served in the army of the potomac.", "head_entity": "union army", "tail_entity": "army of the potomac", "predicted_relation": "part of", "predicted_relation_id": 67}
{"index": 2769, "sentence": "trinigy nor its vista locos did profited by havok in 2011.", "head_entity": "havok", "tail_entity": "trinigy", "predicted_relation": "developer", "predicted_relation_id": 20}
{"index": 2867, "sentence": "subsequently on his career , parker broadcaster the resignation of richard nixon , economic disasters facing the united states , the iran hostage crisis and the attempted assassination of ronald reagan.", "head_entity": "iran hostage crisis", "tail_entity": "united states", "predicted_relation": "location", "predicted_relation_id": 58}
{"index": 2911, "sentence": "the squad owns ago supported canadian army air defense units , but with the retirement of the adats platform this role was discontinued.", "head_entity": "adats", "tail_entity": "canadian army", "predicted_relation": "military branch", "predicted_relation_id": 21}
{"index": 3016, "sentence": "keighley became through residency ( under lawkholme lane ) for the second leg match and drew 5 - 5.", "head_entity": "lawkholme lane", "tail_entity": "keighley", "predicted_relation": "location", "predicted_relation_id": 58}
{"index": 3043, "sentence": "seton ' s 101 dots is the most ever allowed by temple at the liacouras center.", "head_entity": "liacouras center", "tail_entity": "temple", "predicted_relation": "owned by", "predicted_relation_id": 46}
{"index": 3056, "sentence": "scarcely one date before the coincidence did organised , unreported bayern enthusiasts painted parts of the grünwalder stadion , 1860 traditional home stadium , in the club colors of bayern , red and white.", "head_entity": "grünwalder stadion", "tail_entity": "bayern", "predicted_relation": "owned by", "predicted_relation_id": 46}
{"index": 3063, "sentence": "the preliminary limb did made under le colisée in chalon - sur - saône , at the home court of élan chalon.", "head_entity": "le colisée", "tail_entity": "élan chalon", "predicted_relation": "headquarters location", "predicted_relation_id": 32}
{"index": 3073, "sentence": "lingotto represents the site of the oval lingotto , the speed skating venue for the 2006 winter olympics.", "head_entity": "oval lingotto", "tail_entity": "2006 winter olympics", "predicted_relation": "location", "predicted_relation_id": 58}
{"index": 3232, "sentence": "he did mentioned in april 2007 after hyrrokkin , a giantess from norse mythology , who launched hringhorni , baldr ' s funeral ship.", "head_entity": "hringhorni", "tail_entity": "baldr", "predicted_relation": "owned by", "predicted_relation_id": 46}
{"index": 3430, "sentence": "an marvelous appearance against top - flight club arsenal in a 1950 fa cup match resulted in the londoners making an offer for the player , but swansea turned it down.", "head_entity": "1950 fa cup", "tail_entity": "arsenal", "predicted_relation": "participating team", "predicted_relation_id": 3}
{"index": 3458, "sentence": "cska are the advocating russian premier league champions , and will take part in the russian super cup , russian cup and champions league , entering at the group stage.", "head_entity": "russian premier league", "tail_entity": "cska", "predicted_relation": "league", "predicted_relation_id": 52}
{"index": 3492, "sentence": "a small mes than entering apoel , it crowned buckaroo as his team managed to win the cypriot first division title for a fourth time in the row.", "head_entity": "cypriot first division", "tail_entity": "apoel", "predicted_relation": "league", "predicted_relation_id": 52}
{"index": 3496, "sentence": "german jupp heynckes makes the exclusive boy to have earned the cup twice and did so in consecutive years ( 2003 and 2004 ) with german club schalke 04.", "head_entity": "2003", "tail_entity": "schalke 04", "predicted_relation": "participant of", "predicted_relation_id": 10}
{"index": 3589, "sentence": "akon nor giorgio tuinfort co - produced \" mirrors \" in collaboration with kierszenbaum.", "head_entity": "mirrors", "tail_entity": "akon", "predicted_relation": "performer", "predicted_relation_id": 9}
{"index": 3592, "sentence": "it collaborated among billy corgan during the soundtrack to the cinema \" spun \" , singing the movie ' s opening song , a cover of iron maiden \" s \" number of the beast \".", "head_entity": "spun", "tail_entity": "billy corgan", "predicted_relation": "composer", "predicted_relation_id": 51}
{"index": 3811, "sentence": "the duet founded the irrespective docket label psychopathic records with alex abbiss as manager , and produced and starred in the feature films \" big money hustlas \" and \" big money rustlas \".", "head_entity": "big money rustlas", "tail_entity": "psychopathic records", "predicted_relation": "publisher", "predicted_relation_id": 17}
{"index": 3840, "sentence": "the kino was an international hit , and along with \" confession \" proved a breakthrough for anglo - amalgamated.", "head_entity": "confession", "tail_entity": "anglo-amalgamated", "predicted_relation": "publisher", "predicted_relation_id": 17}
{"index": 3889, "sentence": "maalik makes a 2016 pakistani political suspenseful film made by ashir azeem.", "head_entity": "maalik", "tail_entity": "ashir azeem", "predicted_relation": "director", "predicted_relation_id": 18}
{"index": 3895, "sentence": "he did drafted and directed by andjar asmara , a former journalist and stage - writer who had made his directorial debut the preceding year with \" kartinah \".", "head_entity": "kartinah", "tail_entity": "andjar asmara", "predicted_relation": "director", "predicted_relation_id": 18}
{"index": 3902, "sentence": "the roman will feature the historian from khamoshiyan nor two different tales penned by vikram bhatt – 1920 ( film ) and.", "head_entity": "khamoshiyan", "tail_entity": "vikram bhatt", "predicted_relation": "screenwriter", "predicted_relation_id": 55}
{"index": 3909, "sentence": "he did adjusted for the 1962 marathi phraseology replay \" to mee navhech \" , authored by pralhad keshav atre.", "head_entity": "to mee navhech", "tail_entity": "pralhad keshav atre", "predicted_relation": "author", "predicted_relation_id": 17}
{"index": 4179, "sentence": "this quantity was first proposed in the 1990s by british anthropologist robin dunbar , who found a correlation between primate brain size and average social group size.", "head_entity": "robin dunbar", "tail_entity": "anthropologist", "predicted_relation": "occupation", "predicted_relation_id": 36}
{"index": 4216, "sentence": "the cause from the assassinate was revenge for markelov ' s prior work as a lawyer in the interests of anti - russian activists.", "head_entity": "markelov", "tail_entity": "russian", "predicted_relation": "country of citizenship", "predicted_relation_id": 6}
{"index": 4350, "sentence": "for the time of the occupy wall street protests , white was working as a senior editor for \" adbusters \" magazine with occupy wall street co - founder kalle lasn.", "head_entity": "kalle lasn", "tail_entity": "adbusters", "predicted_relation": "publisher", "predicted_relation_id": 17}
{"index": 4431, "sentence": "the senne fer ( ) makes a unaccompanied - trajectory branch line from brackwede to paderborn with a through service to bielefeld in the german state of north rhine - westphalia.", "head_entity": "brackwede", "tail_entity": "bielefeld", "predicted_relation": "place served by transport hub", "predicted_relation_id": 0}
{"index": 4461, "sentence": "underway , the most fundamental airfields are toward lisboa , seaport , faro , funchal ( madeira ) and ponta delgada ( azores ).", "head_entity": "funchal", "tail_entity": "madeira", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
{"index": 4580, "sentence": "also , he owns been named nor parodied per much korean sundry exhibitions , including sbs ' s \" x - man \" and kbs ' s \" star golden bell \".", "head_entity": "star golden bell", "tail_entity": "korean", "predicted_relation": "language of work or name", "predicted_relation_id": 60}
{"index": 4598, "sentence": "onto november 2006 , he took part in the singing competition \" fame x \" ( in hindi language \" chal udiyee \" ) inaugural 2006 - 2007 season.", "head_entity": "fame x", "tail_entity": "hindi language", "predicted_relation": "original language of film or TV show", "predicted_relation_id": 65}
{"index": 4608, "sentence": "for 2010 he appears as a guest artist in the afrikaans improvisational tv - show proesstraat and as judge on the popular afrikaans reality show supersteree alongside shaleen surtie - richards and theuns jordaan.", "head_entity": "proesstraat", "tail_entity": "afrikaans", "predicted_relation": "language of work or name", "predicted_relation_id": 60}
{"index": 4707, "sentence": "karr , whom did initiated off among a lustrous album against cooper ' s ohio state , seemed to have much more trouble beating tressel ' s version of the buckeyes.", "head_entity": "ohio state", "tail_entity": "buckeyes", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
{"index": 4735, "sentence": "haute gave his wta tourism start under the 2007 gastein ladies in both singles and doubles.", "head_entity": "singles", "tail_entity": "2007 gastein ladies", "predicted_relation": "sport", "predicted_relation_id": 68}
{"index": 4757, "sentence": "canadian continued to expand around north america as other british colonies and territories joined with or were ceded to canada , eventually growing from four provinces to ten provinces and three territories.", "head_entity": "colonies", "tail_entity": "north america", "predicted_relation": "located in the administrative territorial entity", "predicted_relation_id": 63}
{"index": 4989, "sentence": "the supplementary railroads toward the peak time throughout eatin nor haltern meanwhile bear this designate , unless this makes never geographically correct because the niers corrientes through mönchengladbach.", "head_entity": "mönchengladbach", "tail_entity": "niers", "predicted_relation": "crosses", "predicted_relation_id": 64}
{"index": 5123, "sentence": "windows eyesight , windows 7 , windows server 2008 and 2008 r2 also support ipp printing over rpc in the \" medium - low \" security zone.", "head_entity": "2008 r2", "tail_entity": "windows server 2008", "predicted_relation": "followed by", "predicted_relation_id": 26}
{"index": 5173, "sentence": "artoustes became licence - created per bristol siddeley ( formerly blackburn ) in the uk , hindustan aeronautics limited in india , and developed by continental cae in the usa as the continental t51.", "head_entity": "bristol siddeley", "tail_entity": "blackburn", "predicted_relation": "said to be the same as", "predicted_relation_id": 50}
{"index": 5174, "sentence": "two typewriters were used for the vocal editing in songs like \" hollywood \" and \" nobody knows me \".", "head_entity": "nobody knows me", "tail_entity": "hollywood", "predicted_relation": "part of", "predicted_relation_id": 67}
{"index": 5190, "sentence": "yoel cellar of tv squad said the no - dating policy felt like a plot contrivance meant strictly to keep ben and leslie apart , which he felt was not good for the story.", "head_entity": "ben", "tail_entity": "leslie", "predicted_relation": "sibling", "predicted_relation_id": 33}
{"index": 5466, "sentence": "shulze did the grandchild of henry muhlenberg and the nephew of brothers peter muhlenberg and frederick muhlenberg , who were leading politicians.", "head_entity": "henry muhlenberg", "tail_entity": "frederick muhlenberg", "predicted_relation": "sibling", "predicted_relation_id": 33}
{"index": 5581, "sentence": "the erect makes featured apparently onto the negotiator ( 1998 ) where it serves as the location of offices for the internal affairs division of the chicago police department.", "head_entity": "the negotiator", "tail_entity": "chicago police department", "predicted_relation": "main subject", "predicted_relation_id": 79}
{"index": 5584, "sentence": "toward 1628 , rupert janson , a faculty through leuven , began typing \" augustinus \" , a three workload pact during antonio from hippopotamus which conflated jesuits among pelagianism per emphasizing augustine ' s propositions.", "head_entity": "augustinus", "tail_entity": "pelagianism", "predicted_relation": "main subject", "predicted_relation_id": 79}
```