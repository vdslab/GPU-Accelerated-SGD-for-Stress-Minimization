# Webアプリの管理範囲

`web/rr-lab/` は、この研究リポジトリの一部として管理します。独立したGitリポジトリやgitlinkは使用しません。リポジトリをcloneすると、サイトのソース・テスト・デプロイ設定も含まれます。

アプリの用途・実行方法・検証記録は `web/rr-lab/README.md` を参照してください。リポジトリ直下から変更を確認できます。

```sh
git status --short
git diff -- web/rr-lab/
```

`node_modules/`、`.wrangler/`、ローカル環境変数ファイルは `web/rr-lab/.gitignore` で除外します。GitHub Pagesへの公開対象は `web/rr-lab/dist/` の静的ファイルのみです。更新は `.github/workflows/pages.yml` で自動公開します。
