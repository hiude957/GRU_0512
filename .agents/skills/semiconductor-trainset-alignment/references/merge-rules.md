# livedata 覆盖窗口内的 110ms 网格合并规则

这一页描述当前项目中已经确定的主流程规则。对于这个 skill，这些规则应按固定标准理解。

当前规则已经从“每天完整 `24h` 输出”调整为“只在 `livedata` 覆盖范围内输出训练/建模网格”。`110ms` 固定网格仍然保留，action 仍然不能插入额外 timestamp 行；区别是 `livedata` 覆盖范围外的 action 只作为隐藏状态更新使用，完成 `state_*` 继承后再从最终训练/输出表中删除。

## 1. 数据检查结论

对当前 `dataset/` 做过只读抽样和全量时间统计：

- `apc`：`1824` 个文件，约 `2726143` 行，采样间隔中位数约 `109ms`，p99 约 `156ms`
- `livedata`：`45` 个文件，约 `1390993` 行，采样间隔中位数约 `795ms`，p99 约 `1466ms`
- `log`：约 `570378` 行，存在大量毫秒级连续 action
- 固定 `110ms` 网格内经常出现多个 action，同一 `io_id` 在同一网格内也会重复出现
- 以当天 `livedata` 覆盖窗口统计，约 `38.40%` 的 action 落在 `livedata` 覆盖范围外或没有当天 `livedata` 的日期上

因此：

- 固定为 `110ms` 网格是可行的，且与 `apc` 的主采样频率接近
- `livedata` 覆盖范围内仍必须被上采样到 `110ms`
- action 不能再通过“插入额外行”处理，必须归并到对应网格
- 同一网格内 action 聚合是必要规则，不是边缘情况
- `livedata` 覆盖范围外的 action 比例不低，不能先删；必须先用它们更新 `state_*`，再过滤最终输出行
- `livedata` 通常只覆盖白天窗口，不应把没有 `livedata` 的夜间理解成设备关闭
- `20260318` 是当前数据集的开机首日：`livedata` 第一条为 `2026-03-18 11:14:23.321`，`log` 第一条为 `2026-03-18 11:14:44.035`，`apc` 最早为 `2026-03-18 08:53:39.088`
- 当前项目口径下，`20260318` 的训练/输出从首条 `livedata` 覆盖开始；首条 `livedata` 前只初始化 `state_*`，不输出 pre-`livedata` 训练行
- 没有 `livedata` 的日期不输出训练行；如果当天有 action，仍要作为隐藏 action 处理，以便后续 `state_*` 继承正确
- 远离 action 的 sensor 变化统计整体很小，但不是严格数学常数；建模上可以把“无 action 时状态基本不变”作为工程假设，同时用 `mask_*` 标明没有真实 sensor

## 2. 总体流程

合并顺序是：

1. 按日历日期连续遍历数据区间，包括没有文件、没有 action、没有 sensor 的日期
2. 把 `apc` 和 `livedata` 都恢复成绝对时间
3. 用 raw `livedata` 划分当天有效覆盖片段；`livedata` 相邻点间隔 `> 2000ms` 时切成两个片段
4. 只在这些 `livedata` 覆盖片段内建立最终训练/输出用的 `110ms` 网格
5. 把 sensor 值投影或插值到输出网格；覆盖片段内保持 `apc_grid > livedata_grid` 的优先级
6. 把全量 `log` action 按 `110ms` 网格归并，包括 `livedata` 覆盖范围外的 action
7. 先用所有 action 按时间顺序更新 `state_*`；覆盖范围外 action 作为隐藏状态更新，不作为最终训练行
8. 在最终保留的 `livedata` 覆盖网格行上展开全部 `evt_*` 与继承后的 `state_*`

最终输出的每一行都必须是 `livedata` 覆盖范围内的固定 `110ms` 网格点，不再因为 action 单独插入额外 timestamp 行。覆盖范围外 action 不能先删，必须在更新 `state_*` 后再过滤掉。

## 3. 时间网格

固定网格步长：

`grid_step_ms = 110`

推荐网格锚点：

- 按自然日 `00:00:00.000` 作为当天 `grid_origin`
- 第 `k` 个网格时间为 `grid_ts_ms = day_start_ms + k * 110`
- 每个网格代表半开区间 `[grid_ts_ms, grid_ts_ms + 110)`
- 最终训练/输出只保留落在当天 `livedata` 有效覆盖片段内的网格点
- 即使某一天没有 `livedata`，也不能在状态机上跳过这一天；如果当天有 action，它仍然负责更新并传递 `state_*` 到下一天
- 当前数据集 `20260318` 的 `grid_origin` 仍按当天 `00:00:00.000` 计算 grid id，但最终训练/输出从首条 `livedata` 覆盖开始，不输出 `00:00:00.000` 到首条 `livedata` 前的 rows

action 归格时使用：

`grid_id = floor((action_ts_ms - day_start_ms) / 110)`

如果 action 正好落在边界上，归入右侧新网格。

`livedata` 覆盖片段定义：

- 用 raw `livedata` 的绝对时间排序
- 相邻 `livedata` 点间隔 `<= 2000ms` 时属于同一有效覆盖片段
- 相邻 `livedata` 点间隔 `> 2000ms` 时切断，不跨这个缺口输出训练网格
- 如果同一天有多个 `livedata` 文件，先转成绝对时间后按时间合并，再按上述规则切片
- 一天没有 `livedata` 时，最终训练/输出行数为 `0`；当天 action 仍作为隐藏状态更新处理

## 4. sensor 时间处理

### `apc`

`apc` 的 `Time` 是相对偏移，必须先换算成绝对时间：

`ts_abs = Process Start Time + Time`

`apc` 的真实采样间隔接近 `110ms`，但不是严格固定，存在 `100ms`、`109ms`、`110ms` 和少量更大间隔。因此不能直接把原始 `apc` 行当作最终时间轴，必须投影到固定网格。

### `livedata`

`livedata` 原本就是绝对时间，直接使用。

`livedata` 明显低于 `110ms` 密度，必须在自身有效覆盖片段内上采样到固定网格。最终训练/输出范围由 `livedata` 覆盖片段决定；不在 `livedata` 覆盖片段内的 `apc` 不单独触发训练/输出行。

## 5. sensor 补值规则

### sensor 维度和原始 mask 质检

当前训练口径固定使用 `150` 个 sensor 数值列：

- `1..148`：正常真实采集的 sensor 维度
- `149..150`：默认补充维度，不作为真实采集 target 或 loss

大多数 raw sensor 行满足固定 mask 模板：

- `mask_1..mask_148 = 1`
- `mask_149..mask_150 = 0`

合并前应先做 raw sensor 行质检：

- 只有满足上述固定 mask 模板的 raw sensor 行，才能作为正常 `apc` / `livedata` 观测参与插值和训练 target
- 不满足该模板的 raw sensor 行视为异常采集行，应从 sensor 候选池中剔除，不参与 `apc_grid`、`livedata_grid`、插值、target 或 loss
- 不需要物理删除 raw 文件；剔除发生在对齐/读取阶段
- 如果某个 APC 文件的所有行都不满足该模板，则该 APC 文件整体从有效 APC 源中排除

当前数据检查发现，`20260416` 有 `34` 个 APC 文件、共 `58106` 行不满足固定 mask 模板；这些行应按异常 sensor 行剔除。剔除后，如果对应时间仍处于 `livedata` 覆盖片段内，应回退使用有效 `livedata_grid`；如果没有有效 `livedata`，该时间点不应用异常 APC 生成真实 sensor target。

### 优先级

在最终保留的 `livedata` 覆盖网格内，sensor 来源优先级：

1. 如果网格点落在有效 `apc` 连续片段内，使用 `apc`
2. 否则，如果网格点落在有效 `livedata` 连续片段内，使用 `livedata`
3. 否则，如果覆盖片段内局部缺失但历史上已经有过可信真实 sensor，使用最近一次可信真实 sensor 做因果前向延续，来源记为 `carried_sensor`
4. 否则，使用 sensor 默认值，来源记为 `sensor_default`

`apc` 覆盖区内不使用 `livedata` 补值。`livedata` 只负责 `livedata` 覆盖片段内的非 `apc` 区域。`livedata` 覆盖片段外不输出 sensor 训练行。

### 连续片段

不要跨长缺口做线性插值。

插值前应先处理重复时间戳：

- 同一来源、同一 timestamp 的重复 sensor 行先归并成一个点
- 对数值 sensor 可取最后一条或取均值，但要固定一种策略并记录
- 对 `mask_*` 建议按列取更保守的有效性；任一重复行该列无效时，该列可置为无效，除非已确认重复行完全一致

推荐阈值：

- `apc` 相邻原始点间隔 `> 250ms` 时，切成两个 `apc` 片段
- `livedata` 相邻原始点间隔 `> 2000ms` 时，切成两个 `livedata` 片段

理由：

- 当前 `apc` 间隔 p99 约 `156ms`，`250ms` 能覆盖正常抖动和少量丢点，同时避免跨过较长缺口造假
- 当前 `livedata` 间隔 p99 约 `1466ms`，`2000ms` 可覆盖大部分正常采集，同时避免跨分钟或跨小时缺口造假
- 当前统计中 `livedata` 相邻间隔 `> 2000ms` 约占 `0.095%`，`apc` 相邻间隔 `> 250ms` 约占 `0.024%`；这两个阈值与当前数据分布匹配

### 数值补值

数值补值和训练有效性要分开理解：

- 输出网格内 sensor 数值列必须尽量填满，保证模型输入矩阵形状稳定
- `mask_*` 或有效性标记决定这个 sensor 是否是真实可信观测
- 对无效补值点，sensor 可以有数值，但对应 `mask_*` 必须置为 `0`，训练 `loss_mask` / `valid_target` 也不能把它当真实 sensor
- `livedata` 覆盖片段外不为了训练输出去补 sensor；如需处理覆盖范围外 action，只保留隐藏 `state_*` 更新，不把这些隐藏行作为 sensor target 或 loss

对普通 sensor 列：

- 网格时间点正好命中原始时间点时，直接使用原始值
- 网格时间点位于同一连续片段的两个原始点之间时，做线性插值，并保留真实 `mask_*`
- 网格时间点位于两个真实 sensor 点之间，但不满足有效插值边界时，仍允许为了输入形状做线性插值，但对应 `mask_*` 必须置为 `0`
- 网格时间点只有左侧真实 sensor 点时，做因果前向延续，来源为 `carried_sensor`，并将对应 `mask_*` 置为 `0`
- 网格时间点只有右侧真实 sensor 点、但没有任何历史真实 sensor 时，训练和推理输入不能用未来 sensor 回填，应使用 `sensor_default`，并将对应 `mask_*` 置为 `0`
- 左右两侧都没有任何真实 sensor 时，优先继承历史 carried sensor；如果历史上也没有真实 sensor，再使用 sensor 默认值，并将对应 `mask_*` 置为 `0`

### `carried_sensor` 与 `sensor_default`

`carried_sensor` 表示当前 `110ms` 网格没有真实 sensor，但此前已经出现过可信 `apc` 或 `livedata`，所以用最近一次可信真实 sensor 做前向延续。

- `carried_sensor` 是部署一致的因果填充方式
- 在最终输出被限制到 `livedata` 覆盖片段后，夜间没有 `livedata` 的长区间通常不再输出 `carried_sensor` rows
- action 发生在当天 `livedata` 覆盖范围外时，应作为隐藏 action 更新 `state_*`；不需要为了最终训练表输出 carried sensor 行
- `carried_sensor` 的 `mask_*` 必须为 `0`，不能作为 sensor target 或 loss

`sensor_default` 表示当前网格既没有真实 sensor，也没有任何历史真实 sensor 可继承，只能使用默认初始 sensor。

- `sensor_default` 主要发生在整个序列第一天开始、第一次开机、或从中途恢复但没有上一天 sensor checkpoint 的场景
- 对当前数据集，`20260318 00:00:00.000` 到 `2026-03-18 11:14:23.321` 首条 `livedata` 之前属于初始化阶段；初始化 `state_*`，但不输出 pre-`livedata` 训练 rows。即使该区间内存在早于 `livedata` 的 `apc` 文件，也不把它作为训练 target 或可信建模起点
- 不要每天重新使用 `sensor_default`
- 当前 `gas_header.xlsx` 更像 sensor 类型表，不是完整默认值表；如果没有明确的 sensor 默认值列或独立 `sensor_default.xlsx`，应使用统一默认值，并保持 `mask_* = 0`
- 一旦出现第一条可信真实 sensor，后续无真实 sensor 的网格应转为 `carried_sensor`

离线核对表可以额外记录 `invalid_linear_fill` 或 `invalid_backfill` 这类填充方式，但它们不能作为部署一致的 GRU 输入依据。训练输出限制到 `livedata` 覆盖片段后，应尽量减少 `carried_sensor` / `sensor_default` 行；如果保留它们，只能作为 `mask_* = 0` 的输入上下文，不能作为 sensor target 或 loss。

### `apc` 插值边界

对某个 `110ms` 网格点 `grid_ts`，如果它被两个 `apc` 原始点夹住：

`t_left <= grid_ts <= t_right`

计算：

- `gap = t_right - t_left`
- `nearest = min(grid_ts - t_left, t_right - grid_ts)`

只有同时满足以下条件时，才使用 `apc` 线性插值：

- `gap <= 250ms`
- `nearest <= 125ms`

插值公式：

`sensor(grid_ts) = sensor_left + (grid_ts - t_left) / (t_right - t_left) * (sensor_right - sensor_left)`

如果 `grid_ts` 距离最近 `apc` 点约 `50ms`，且左右 `apc` 点属于同一连续片段，则这是正常情况，应使用线性插值。

如果不满足 `apc` 有效插值条件，可以继续用左右 `apc` 点做离线数值线性补值，但该补值不算真实 sensor，对应 `mask_*` 必须置为 `0`。训练和推理输入中，如果缺少左侧 `apc` 点，应退到历史 `carried_sensor` 或 `sensor_default`；如果缺少右侧 `apc` 点，应使用左侧最近可信真实 sensor 做 `carried_sensor`。

无效 `apc` 数值补值只用于维持输入形状，不参与 sensor target 或 loss。

### `livedata` 插值到 `110ms`

`livedata` 不要先插成中间频率再二次插值。应直接从 raw `livedata` 插值到最终 `110ms` 网格。

对某个不在有效 `apc` 片段内的 `110ms` 网格点 `grid_ts`，找到夹住它的两个 `livedata` 原始点：

`t_left <= grid_ts <= t_right`

计算：

`gap = t_right - t_left`

只有满足以下条件时，才使用 `livedata` 线性插值：

- `gap <= 2000ms`

插值公式：

`sensor(grid_ts) = sensor_left + (grid_ts - t_left) / (t_right - t_left) * (sensor_right - sensor_left)`

如果 `gap > 2000ms`，不要跨这个缺口输出训练网格。离线核对表可以跨缺口做数值线性补值并标为无效，但训练/输出表应按 `livedata` 片段切断。

如果 action 发生在 `livedata` 覆盖片段之前、之后、或两个 `livedata` 片段之间，应作为隐藏 action 更新 `state_*`。这些 action 不应使用未来 sensor 回填，也不应作为最终训练/输出 rows 保留。

最终优先级始终是：

`apc_grid` 有效插值 > `livedata_grid` 有效插值 > `carried_sensor` > `sensor_default`

对 `mask_*` 列：

- 不做线性插值
- 正常真实采集行使用固定模板：`mask_1..mask_148 = 1`，`mask_149..mask_150 = 0`
- 异常 raw mask 行已在合并前剔除，不应把异常 mask 传播到最终训练表
- 如果左侧没有真实点，不能为了训练输入使用未来 `mask_*` 伪装成有效观测
- 无效线性补值、`carried_sensor`、`sensor_default` 的对应 `mask_*` 必须置为 `0`

对最终输出范围内没有可信 sensor 支撑的网格：

- 可以为了保持矩阵形状填 sensor 数值，包括离线无效线性补值、`carried_sensor` 或 `sensor_default`
- 不应把这些补出来的值当作真实 sensor
- 对应 `mask_*` 或有效性标记必须置为无效
- 训练 target 和 loss 不应使用这些无效 sensor 点

## 6. action 网格聚合

`log` 中每一行代表一次 action：

- `timestamp`
- `io_id`
- `io_value`

所有 action 都归并到固定 `110ms` 网格中，不再插入独立 action 行。这里的“所有 action”包括 `livedata` 覆盖范围外的 action；它们必须先参与 `state_*` 更新，再根据是否落在最终输出范围内决定是否保留该行。

聚合规则：

- 同一网格内不同 `io_id` 的 action 要叠加保留
- 同一网格内同一 `io_id` 出现多次 action 时，只取该网格内最后发生的一条作为该 `io_id` 的最终状态
- 如果网格落在最终 `livedata` 覆盖输出范围内，`evt_{id}` 表示该网格内该 `io_id` 是否发生过 action，发生过则为 `1`
- 如果网格落在最终输出范围外，该 action 只更新隐藏 `state_{id}`，最终训练/输出表不保留对应 `evt_*` 行
- `state_{id}` 使用该网格内该 `io_id` 最后一次 action 的 `io_value`，并从该网格之后持续继承

同一网格内 action 的先后顺序只用于确定同一 `io_id` 的最后状态。不同 `io_id` 之间不需要排序后展开成多行。

## 7. `evt_*` 与 `state_*`

每个全局 `io_id` 都展开两列：

- `evt_{id}`：该网格内该 IO 是否发生 action
- `state_{id}`：该网格结束后该 IO 的持续状态

规则如下：

- `evt_*` 是网格级瞬时事件，发生为 `1`，否则为 `0`
- `state_*` 不是只在 action 当下有效，而是会持续继承
- 某个 `io_id` 在当前网格发生 action 后，从当前网格开始使用新 `state_*`
- 同一网格内同一 `io_id` 多次 action 时，只保留最后一次 `io_value`
- 从未出现过的 `io_id`，初始状态按默认值；没有默认值时按 `0`
- 状态既跨网格继承，也跨天继承
- 整个数据集第一天开始时，`state_*` 从 `dataset/action_default.xlsx` 初始化
- 对当前数据集，`20260318 00:00:00.000` 到首条 `livedata` 之前，`state_*` 使用 `action_default.xlsx` 的初始状态；如果该区间出现 action，则作为隐藏 action 更新后续 `state_*`，但不输出 pre-`livedata` 训练行
- 如果从中途恢复训练或推理，并且存在上一天状态 checkpoint，应优先使用 checkpoint；只有没有 checkpoint 时才回退到 `action_default.xlsx`
- 空日不代表设备关闭。没有 shutdown/reset 证据时，空日的 `state_*` 全部从上一天持续继承到下一天；没有 `livedata` 的日期不输出训练行，但当天 action 仍会更新继承状态

## 8. `source` 建议

固定网格后，`source` 应表达 sensor 值的来源，而不是表达是否有 action。最终训练/输出只保留 `livedata` 覆盖范围内 rows，覆盖范围外隐藏 action 不需要写入最终 `source`。

推荐来源：

- `apc_grid`：该网格 sensor 来自 `apc` 原始点或 `apc` 片段内插值
- `livedata_grid`：该网格 sensor 来自 `livedata` 原始点或 `livedata` 片段内插值
- `carried_sensor`：最终输出范围内局部没有真实 sensor 时，用最近一次可信真实 sensor 做因果前向延续
- `sensor_default`：最终输出范围内既没有真实 sensor，也没有历史 sensor 可继承时，只能使用默认初始 sensor
- `missing_sensor`：可作为无真实 sensor 的总称，但训练表中更推荐拆成 `carried_sensor` 与 `sensor_default`

最终输出范围内是否有 action 应由 `evt_*`、`has_action` 或 `action_count` 表达，不应再通过 `logapc`、`loglivedata`、`logonly` 这类 source 名称表达。最终输出范围外的 action 不作为训练行保留，但它们的状态更新必须已经体现在后续保留行的 `state_*` 中。

如果下游仍依赖旧的 `source_code`，需要同步扩展映射表，不要继续假设 `source_code` 永远只有 `apc/livedata/logapc/loglivedata` 四类。
