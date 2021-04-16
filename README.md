# quantum_computing_memo
量子コンピュータ関連のメモやサンプルを配置するリポジトリ。計算用のノートブックや、サンプルアプリを登録する。

## jupyter notebookについて

jupyter notebookはGithubで見れるものの、Texが崩れるため可読性が悪い。PDFに変換して登録する。

変換方法については下記が参考になった。
https://qiita.com/masa-ita/items/fb61263cd49cf949b1bc

上記のサイトをもとに、Dockerfileに環境を整えた。日本語ノートブックに対応済み。(tools/ipynb2pdf/dockerfile)
イメージをビルドし、コンテナ内で下記のようにすればPDF化できる。

```
# jupyter nbconvert --to pdf glover計算.ipynb
```

PDF化する際の記載上の注意点

* 数式環境として$を2つ使う方法とbegin{align}などを使う方法がある。$の中でbegin{align}をするとエラーになるので注意（Texに慣れている人からすれば当たり前だと思うが、私は久々で引っかかった）
* defを行うことで、Texのマクロを定義できるが、下記のダブルスタンダードな状態である。
  * $の中でdefすることでjupyter上の数式に反映される。
  * $の外でdefすることでPDF化する際に反映される。（PDF化するときにTexを経由するが、数式環境の中ではdefすることができず、未定義エラーになる）
  * このためjupyter notebookではマクロを$内に入れたバージョンと外に出したバージョンの2つを定義しておくべし。