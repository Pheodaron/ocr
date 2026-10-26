import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

test("Проект → документ → LightOn → визуальная правка → утверждённый набор", async ({
  page,
}) => {
  await page.goto("/");
  page.once("dialog", (dialog) => dialog.accept("Проверка таблиц"));
  await page
    .getByRole("button", { name: "Создать проект", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Проверка таблиц", exact: true }),
  ).toBeVisible();
  const png = readFileSync(new URL("./fixtures/table.png", import.meta.url));
  await page
    .locator("input[type=file]")
    .first()
    .setInputFiles({ name: "таблица.png", mimeType: "image/png", buffer: png });
  await expect(
    page.getByRole("button", { name: "таблица.png", exact: true }),
  ).toBeVisible();
  await expect(page.locator(".documents-table")).toContainText("1 / 1", {
    timeout: 15000,
  });
  await page.getByRole("button", { name: "1. Распознать LightOn" }).click();
  await expect(
    page.locator(".job-row").filter({ hasText: "LightOn 4B" }),
  ).toContainText("Завершено", { timeout: 15000 });
  await page.getByRole("button", { name: "таблица.png", exact: true }).click();
  await expect(page.locator(".markdown")).toContainText("0012");
  await page.getByRole("button", { name: "Таблицы", exact: true }).click();
  await page.getByRole("textbox", { name: "Строка 2, столбец 1" }).fill("0013");
  await page
    .getByRole("button", { name: "Утвердить текст и таблицы", exact: true })
    .click();
  await expect(page.getByRole("status")).toContainText(
    "Текст и таблицы утверждены",
  );
  await page.getByRole("button", { name: "Документы", exact: true }).click();
  await page
    .getByRole("checkbox", { name: "Выбрать таблица.png", exact: true })
    .check();
  await page
    .getByRole("button", { name: "Обучение", exact: true })
    .last()
    .click();
  await page
    .getByRole("button", { name: "Наборы данных", exact: true })
    .click();
  page.once("dialog", (dialog) => dialog.accept("Первая версия"));
  await page
    .getByRole("button", { name: "Создать версию", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Первая версия" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Посмотреть эталоны" }).click();
  await expect(page.locator(".markdown")).toContainText("0013");
  await page.screenshot({ path: "test-results/workbench.png", fullPage: true });
});
