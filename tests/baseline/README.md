# Step 6-2：モジュール化前の基準データ

`main.py` を分割した後も、現在の数値計算結果を維持できているか確認するための基準です。既存の計算コードは変更せず、実際の実行から取得しました。

## Step 6-11のチェックポイント

`reference/` はStep 6-2で取得した基準のまま固定しています。Step 6-3〜6-10では、運動学を `robot_arm/kinematics.py`、整定時間を `metrics.py`、PID・RK2を `simulation.py`、アニメーション更新を `visualization.py` へ移動しました。履歴の遅延取得と `FuncAnimation` の設定は `main.py` に残しています。

Gitには、この基準データ1組、`capture.py`、`compare.py`、本説明、取得時の `verification.json`、最新の `step_6_10_gui_verification.json`、再実行結果の `checkpoint_verification.json` を保存します。重複する `candidate*/` と過去の各Stepの診断記録はローカル用として除外します。候補データは、下記のコマンドで新しいフォルダーに再取得できます。

`.gitattributes` で `reference/` 配下の改行変換を無効にし、Gitへ保存する際も取得済みファイルのバイト列とSHA-256を維持しています。基準データの再生成や改行の書き換えは行っていません。

`checkpoint_verification.json` に構文・数値・描画・出力の照合結果と対象ソースのハッシュを記録しています。TkAggの非表示ウィンドウで全101フレームを更新した結果は `step_6_10_gui_verification.json` に保存しています。**通常ウィンドウの目視再生、対話操作、既存の前面表示操作は未検証です。**

## 基準となる環境

| 項目 | 記録値 |
| --- | --- |
| Git commit | `6788e27933cbf79a8565c657a391b47c2b464ad3` |
| ブランチ | `master` |
| Python | 3.13.17 |
| NumPy | 2.5.4 |
| Matplotlib | 3.11.2 |
| OS | Windows 11 |
| 基準取得時の描画バックエンド | Agg（GUIなし） |

`reference/environment.json` に基準取得時の実行日時（UTC）、実行ファイル、設定値、取得位置、Git情報、ファイルのSHA-256も保存しています。基準取得時の `source_matches_commit` は `true` です。当時は取得用ファイルが未コミットだったため、取得時のGit状態には `tests/` が未追跡として記録されています。この来歴情報も取得時のまま維持しています。

## 保存したファイル

| ファイル | 内容 |
| --- | --- |
| `capture.py` | 元の `main.py` を実行し、新しい出力フォルダーへ基準形式で保存する取得ツール |
| `compare.py` | 2組のデータを読み取り、構造・数値・描画情報を比較するツール |
| `verification.json` | 独立した再実行との完全一致、差の検出、上書き拒否などの検証記録 |
| `checkpoint_verification.json` | Step 6-11で再実行した回帰テスト、保護対象のハッシュ、GUI検証範囲の記録 |
| `step_6_10_gui_verification.json` | TkAggの非表示ウィンドウで全101フレームを更新した記録。目視確認は未実施 |
| `reference/environment.json` | commit、環境、取得方法、設定値、保護対象・生成物のハッシュ |
| `reference/summary.json` | 丸め前の主要指標、整定時間、ゲイン探索結果、RK2実験結果 |
| `reference/arrays.npz` | 241個の配列。誤差・角度・速度・PID状態・目標値・描画用データなど |
| `reference/arrays_manifest.json` | 各配列のキー、形状、dtype、単位、取得元 |
| `reference/plots.json` | 16図のサイズ、軸、ラベル、凡例、線種、色、パッチなどの描画情報 |
| `reference/stdout.txt` | `main.py` が実行中に出力した標準出力 |
| `reference/stderr.txt` | `main.py` 実行中の標準エラー出力。今回の取得では空 |
| `reference/trajectory_tracking_comparison.png` | 元の保存処理の出力先をこのフォルダーへ振り替えて生成した比較画像 |

NPZは元の配列のdtypeを保ち、pickleを使用せずに保存しています。JSONの数値にも表示用の丸めは行っていません。以下の表だけは読みやすさのため小数点以下4桁で表示しています。

## 追従・保持の基準結果

両方式ともPID＋フィードフォワード制御で追従し、1秒以降はPIDのみで保持します。

| 指標 | 等速軌道 | 5次多項式軌道 |
| --- | ---: | ---: |
| 最大追従位置誤差 | 13.6747 mm | 0.3029 mm |
| 平均追従位置誤差 | 7.4047 mm | 0.1647 mm |
| 1秒時点の位置誤差 | 0.0537 mm | 0.2672 mm |
| 1秒時点の先端速度の大きさ | 113.5824 mm/s | 0.1818 mm/s |
| 保持中の最大位置誤差 | 15.9937 mm | 0.2672 mm |
| 2秒時点の位置誤差 | 1.7713 mm | 0.0216 mm |

追従区間は0〜1秒、保持区間は1〜2秒です。それぞれ端点を含む21点を記録しています。平均値は追従21点の算術平均、最大値は記録点での最大値です。

等速側と5次側ではFF用角加速度の計算方法も異なります。この表は、その違いを含む現在の実験条件での基準結果です。

追加の主要結果も `summary.json` に保存しています。

| 指標 | 基準値（表示用） |
| --- | ---: |
| 角速度指令型P制御の第2関節整定時間 | 1.90 s |
| 固定目標・最適化前PIDの第2関節整定時間 | 5.05 s |
| 固定目標・最適化後PIDの第2関節整定時間 | 4.55 s |
| 固定目標の先端位置整定時間（20 mm） | 7.70 s |
| 固定目標の最終位置誤差 | 10.4813468950 mm |
| ゲイン探索で選ばれた `(Kp, Ki, Kd)` | `(1.2, 0.3, 0.4)` |

`summary.json` の整定時間が `null` の項目は、実行した範囲では整定しなかったという元の計算結果です。推測値や仮の値による補完はしていません。

## 丸め前の履歴と状態

配列キーの一覧と単位は `arrays_manifest.json` を参照してください。主なグループは次のとおりです。

- `kinematic_p.*`：角速度指令型P制御の時刻・角度履歴。
- `fixed_target.*`：P・PD・PID、最適化前後、先端誤差、ヤコビ行列による予測値。
- `constant.*`：等速軌道の目標値、PIDのみ／PID＋FFの誤差、追従・保持の状態。
- `quintic.*`：5次軌道の目標位置・速度・加速度、関節目標値、追従・保持の状態。
- `comparison.constant.*`：Step 5-22で再計算した等速側の比較データ。
- `gain_sweep.run_*`：`simulate_pid()` の56回の呼び出しで得た角度履歴。引数と最終状態は `summary.json` に保存。
- `rk2.run_*`：`simulate_rk2()` の3回の呼び出しの最終角度。時間刻みと最終誤差は `summary.json` に保存。
- `plot.*`：各図の線データ、散布点、パッチの形状など。
- `animation.frame_*`：元の `update()` をフレーム0・20・100・200で呼び出して得たアーム座標。

`*.state_history.*` は、対応する時刻の制御更新前の状態です。追従・保持とも21点になり、保持の最終点はループ終了時の状態を追加しています。角度・角速度だけでなく、積分誤差（`rad*s`）と前回誤差も記録しています。

対応する時刻列は、等速追従が `constant.time_s`、等速保持が `constant.hold_time_s`、5次追従が `quintic.time_s`、5次保持が `quintic.hold_time_s`、再比較が `comparison.constant.time_s` です。

追従終了時と保持終了時の状態は別々にコピーしています。保持による配列の更新や、後続処理による変数の再利用で基準値が失われないようにしています。

読み取り例：

```python
import numpy as np

with np.load("tests/baseline/reference/arrays.npz", allow_pickle=False) as data:
    time = data["constant.time_s"]
    error = data["constant.pid_ff_error_mm"]
    theta = data["quintic.pid_ff.state_history.theta_rad"]
```

## 実行方法と既存ファイルの保護

取得ツールは、元の `main.py` を `runpy.run_path()` で実行します。計算式、ゲイン、時間刻み、ループの順序は置き換えていません。行・関数終了イベントを観測し、必要な状態や戻り値をコピーしています。

この取得プロセス内に限り、次の表示・出力操作を変更します。

- MatplotlibをAggに設定し、`plt.show()` を無効化。
- Tk固有の `window.lift()` と `window.attributes()` を何もしない操作に置換。
- `Figure.savefig()` の保存先を、新しく指定した出力フォルダーへ変更。

取得処理は `main.py` 自体を書き換えません。`main.py`、ルートの `README.md`、既存の `assets/` 内のファイルについて実行前後のハッシュ一致を確認しています。元の標準出力にある `Saved graph: assets/...` は元スクリプトの表示で、実際の振り替え先は `environment.json` に記録しています。

今回、GUI操作は必要ありませんでした。GUIの前面表示、対話操作、アニメーションの実時間での再生は検証対象に含めていません。比較PNGは既存画像と同じSHA-256になりました。

## 今後の比較手順

プロジェクトのルートからPowerShellで実行します。`reference/` は固定し、変更後の結果は別の新規フォルダーへ保存してください。既に存在する出力フォルダーへの上書きは拒否します。

```powershell
.\.venv\Scripts\python.exe tests\baseline\capture.py --output tests\baseline\candidate
.\.venv\Scripts\python.exe tests\baseline\compare.py tests\baseline\reference tests\baseline\candidate --exact
```

同じ環境でファイル移動だけを行う段階では、まず `--exact` による完全一致を確認します。許容誤差を使う場合は、次のように実行します。

```powershell
.\.venv\Scripts\python.exe tests\baseline\compare.py tests\baseline\reference tests\baseline\candidate --rtol 1e-12 --atol 1e-12
```

比較ツールは次を確認します。

1. 両方の取得が `status: complete` で完了していること。
2. 配列キー、形状、dtypeが一致し、NaNや無限大が含まれないこと。
3. 全配列と主要指標の数値が一致すること。差がある配列は最大絶対差を表示。
4. 図のタイトル、軸、凡例、線種などの情報が一致すること。

成功時は終了コード0、差や読み取りエラーがある場合は1です。既定の許容誤差は `rtol=1e-12`、`atol=1e-12` で、各配列の保存単位に適用します。bool配列は常に完全一致で比較します。

commit、ソースのハッシュ、取得日時などの来歴情報は数値の比較条件にしていません。環境差は通知し、ライブラリのバージョンが異なる場合は実際のバージョンも表示します。数値が一致しても、環境差の内容は確認してください。標準出力の全文やPNGのバイト列は自動比較せず、PNGは目視確認用にも残しています。

取得ツールは `main.py` の実験変数・処理位置と、`robot_arm/simulation.py` の関数終了時の値を観測します。アニメーションは `main.py` の `animation_update` を使用し、旧構成の `update` にも対応しています。今後これらの構成が変わる段階では、候補側の取得処理を新しい戻り値へ対応させ、同じ配列キー・単位・形状で出力してください。計算処理を取得ツールに複製せず、変更後のシミュレーションが実際に計算した値を取り出します。取得位置が見つからない場合はエラーとして停止し、欠けた値を推測で作りません。

Step 6-2の確認では、独立した2回の実行について、241個の配列・主要指標・描画情報が `--exact` で一致しました。検証用の一時データに0.01 mmの差を加えると比較が失敗すること、既存の基準フォルダーへの上書きが拒否されることも確認しています。検証用の一時フォルダーは削除し、結果を `verification.json` に保存しました。Step 6-11での再実行結果は `checkpoint_verification.json` を参照してください。
